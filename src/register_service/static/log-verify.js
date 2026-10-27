// The Guild Register's log, checked in the browser: the same algorithms as khaos_attribution.translog (RFC 9162).
"use strict";
(function () {
  const form = document.getElementById("log-form");
  const out = document.getElementById("log-result");
  const input = document.getElementById("log-input");
  if (!form) return;
  const base = form.dataset.mirror + "/log/";
  const enc = new TextEncoder();

  const hex = (b) => Array.from(new Uint8Array(b), (x) => x.toString(16).padStart(2, "0")).join("");
  const unhex = (h) => new Uint8Array(h.match(/../g).map((x) => parseInt(x, 16)));
  const isHex = (h) => typeof h === "string" && /^[0-9a-f]{64}$/.test(h);
  const sha = async (bytes) => new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  const cat = (...parts) => {
    const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let i = 0;
    for (const p of parts) { all.set(p, i); i += p.length; }
    return all;
  };
  const leafHash = (bytes) => sha(cat(new Uint8Array([0]), bytes));
  const nodeHash = (l, r) => sha(cat(new Uint8Array([1]), l, r));
  const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

  // RFC 8785: ECMAScript number and string forms, keys sorted by UTF-16 code units.
  function jcs(v) {
    if (v === null || typeof v !== "object") return JSON.stringify(v);
    if (Array.isArray(v)) return "[" + v.map(jcs).join(",") + "]";
    return "{" + Object.keys(v).sort().map((k) => JSON.stringify(k) + ":" + jcs(v[k])).join(",") + "}";
  }

  async function verifyInclusion(leaf, index, size, path, root) {
    if (!(index >= 0 && index < size)) throw new Error("the leaf index is outside the tree");
    let fn = index, sn = size - 1, r = leaf;
    for (const p of path) {
      if (sn === 0) throw new Error("the inclusion proof is longer than the tree");
      if (fn & 1 || fn === sn) {
        r = await nodeHash(p, r);
        if (!(fn & 1)) while (!(fn & 1) && fn !== 0) { fn >>= 1; sn >>= 1; }
      } else {
        r = await nodeHash(r, p);
      }
      fn >>= 1; sn >>= 1;
    }
    if (sn !== 0) throw new Error("the inclusion proof is shorter than the tree");
    if (!same(r, root)) throw new Error("the inclusion proof does not lead to the head's root");
  }

  async function verifyConsistency(first, second, firstRoot, secondRoot, proof) {
    if (first === second) {
      if (proof.length || !same(firstRoot, secondRoot)) throw new Error("two heads of one size disagree");
      return;
    }
    const path = proof.slice();
    if ((first & (first - 1)) === 0) path.unshift(firstRoot);
    let fn = first - 1, sn = second - 1;
    while (fn & 1) { fn >>= 1; sn >>= 1; }
    let fr = path[0], sr = path[0];
    for (const c of path.slice(1)) {
      if (sn === 0) throw new Error("the consistency proof is longer than the tree");
      if (fn & 1 || fn === sn) {
        fr = await nodeHash(c, fr); sr = await nodeHash(c, sr);
        if (!(fn & 1)) while (!(fn & 1) && fn !== 0) { fn >>= 1; sn >>= 1; }
      } else {
        sr = await nodeHash(sr, c);
      }
      fn >>= 1; sn >>= 1;
    }
    if (sn !== 0 || !same(fr, firstRoot) || !same(sr, secondRoot)) throw new Error("the heads are not consistent: the log was rewritten");
  }

  async function get(name, raw) {
    const r = await fetch(base + name, { cache: "no-cache" });
    if (r.status === 404) return null;
    if (!r.ok) throw new Error(name + ": " + r.status);
    return raw ? new Uint8Array(await r.arrayBuffer()) : r.json();
  }

  function pemBytes(pem) {
    const b64 = pem.replace(/-----[^-]+-----/g, "").replace(/\s+/g, "");
    return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  }

  async function keyIdOf(pem) {
    const der = pemBytes(pem);
    return hex(await sha(der.slice(der.length - 32))).slice(0, 16);
  }

  // A head's signature covers exactly the RFC 8785 form of these five fields.
  async function checkHead(head, pem) {
    if (head.log_id !== "guild-register" || head.version !== "guild.register.log.head/1") throw new Error("not a head of this log");
    if (head.key_id !== (await keyIdOf(pem))) throw new Error("the head is signed by key " + head.key_id + ", not the pinned one");
    const payload = { log_id: head.log_id, root_sha256: head.root_sha256, timestamp: head.timestamp, tree_size: head.tree_size,
      version: head.version };
    let key;
    try {
      key = await crypto.subtle.importKey("spki", pemBytes(pem), { name: "Ed25519" }, false, ["verify"]);
    } catch (e) {
      return false;
    }
    const ok = await crypto.subtle.verify({ name: "Ed25519" }, key, unhex(head.signature), enc.encode(jcs(payload)));
    if (!ok) throw new Error("the head's signature is not the log key's");
    return true;
  }

  async function resolve(text) {
    const t = text.trim();
    if (isHex(t.toLowerCase())) {
      const h = t.toLowerCase();
      const hashes = await get("hashes.json");
      if (hashes && hashes.leaves.includes(h)) return [h, "the leaf hash given"];
      return byRecord(h);
    }
    let doc;
    try { doc = JSON.parse(t); } catch (e) { throw new Error("not JSON and not a sha256 hex digest"); }
    if (doc && doc.version === "guild.register.log.entry/1") return [hex(await leafHash(enc.encode(jcs(doc)))), "the entry pasted, hashed here"];
    if (doc && isHex(doc.record_sha256)) return byRecord(doc.record_sha256, true);
    throw new Error("neither a log entry nor a register record");
  }

  async function byRecord(h, fromDoc) {
    const records = (await get("records.json")) || {};
    const admit = (records[h] || []).find((x) => x.kind === "admit");
    if (!admit) throw new Error("no admitted entry for record " + h);
    return [admit.leaf_hash, fromDoc ? "the record's hash as the pasted document states it; the command also rehashes a full signed record" : "the record's admit entry"];
  }

  async function included(leafHex, latest, pem) {
    const proof = await get("proofs/" + leafHex + ".json");
    if (!proof) throw new Error("the mirror holds no proof for " + leafHex);
    let head = latest, signed = true;
    if (proof.tree_size !== latest.tree_size) {
      head = await get("sth/" + proof.tree_size + ".json");
      signed = await checkHead(head, pem);
    }
    await verifyInclusion(unhex(leafHex), proof.leaf_index, head.tree_size, proof.audit_path.map(unhex), unhex(head.root_sha256));
    return [proof, head, signed];
  }

  function show(lines) {
    out.replaceChildren(...lines.map(([cls, text]) => { const p = document.createElement("p"); if (cls) p.className = cls; p.textContent = text; return p; }));
  }

  async function run(text) {
    show([["", "Checking…"]]);
    const keyDoc = await get("key.json");
    const pem = form.dataset.pinnedPem || (keyDoc && keyDoc.public_key_pem);
    if (!pem) throw new Error("no log key to check against");
    const latest = await get("sth/latest.json");
    if (!latest) throw new Error("the log has no signed head yet");
    let signed = await checkHead(latest, pem);
    const [leafHex, how] = await resolve(text);
    const [proof, head, headSigned] = await included(leafHex, latest, pem);
    signed = signed && headSigned;
    if (head !== latest) {
      const c = await get("consistency/" + head.tree_size + ".json");
      if (!c || c.second !== latest.tree_size) throw new Error("the mirror changed while it was read; try again");
      await verifyConsistency(head.tree_size, latest.tree_size, unhex(head.root_sha256), unhex(latest.root_sha256), c.proof.map(unhex));
    }
    const body = await get("entries/" + leafHex + ".json", true);
    let entry = null;
    if (body) {
      if (hex(await leafHash(body)) !== leafHex) throw new Error("the entry's body does not hash to its leaf");
      entry = JSON.parse(new TextDecoder().decode(body));
    }
    const struck = ((await get("struck.json")) || {})[leafHex];
    if (struck) await included(struck.leaf_hash, latest, pem);
    const what = entry ? entry.kind + " entry" + (entry.serial ? " for No. " + entry.serial : "") : "entry (its body is withheld)";
    const lines = [["ok", "Proved: the " + what + " is leaf " + proof.leaf_index + " of the log at size " + head.tree_size + "."],
      ["", "Found by " + how + ". Head signed " + head.timestamp + " by log key " + head.key_id + "."]];
    if (!signed) lines.push(["bad", "This browser cannot check Ed25519 signatures: the proof holds, but check the head's signature with khaos-log-verify."]);
    if (!form.dataset.pinnedPem) lines.push(["bad", "The key is the one the register serves, not one committed in the contract."]);
    if (struck) lines.push(["bad", "Struck: " + (struck.reason || "") + (struck.sitting ? " (" + struck.sitting + ")" : "")]);
    if (entry) lines.push(["", jcs(entry)]);
    show(lines);
  }

  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    run(input.value).catch((e) => show([["bad", "Not proved: " + e.message]]));
  });
  if (location.hash.length === 65) {
    input.value = location.hash.slice(1);
    form.requestSubmit();
  }
})();
