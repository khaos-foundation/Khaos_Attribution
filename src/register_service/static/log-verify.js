// The Guild Register's log, checked in the browser: the same algorithms and rules as khaos_attribution.translog.verify.
"use strict";
(function () {
  const form = document.getElementById("log-form");
  const out = document.getElementById("log-result");
  const input = document.getElementById("log-input");
  if (!form) return;
  const base = form.dataset.mirror + "/log/";
  const enc = new TextEncoder();
  const STALE_HOURS = 48;
  const DELAY_ALLOWANCE_DAYS = 9;

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
  const hours = (earlier, later) => (Date.parse(later) - Date.parse(earlier)) / 3600000;

  // RFC 8785: ECMAScript number and string forms, keys sorted by UTF-16 code units.
  function jcs(v) {
    if (v === null || typeof v !== "object") return JSON.stringify(v);
    if (Array.isArray(v)) return "[" + v.map(jcs).join(",") + "]";
    return "{" + Object.keys(v).sort().map((k) => JSON.stringify(k) + ":" + jcs(v[k])).join(",") + "}";
  }

  // RFC 6962's tree hash: split at the largest power of two below the size.
  async function treeHash(leaves) {
    if (leaves.length === 1) return leaves[0];
    let k = 1;
    while (k * 2 < leaves.length) k *= 2;
    return nodeHash(await treeHash(leaves.slice(0, k)), await treeHash(leaves.slice(k)));
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

  async function get(name) {
    const r = await fetch(base + name, { cache: "no-cache" });
    if (r.status === 404) return null;
    if (!r.ok) throw new Error(name + ": " + r.status);
    return r.json();
  }

  function pemBytes(pem) {
    const b64 = pem.replace(/-----[^-]+-----/g, "").replace(/\s+/g, "");
    return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  }

  async function keyIdOf(pem) {
    const der = pemBytes(pem);
    return hex(await sha(der.slice(der.length - 32))).slice(0, 16);
  }

  // A head's signature covers exactly the RFC 8785 form of these five fields; false when the browser has no Ed25519.
  async function checkHead(head, pem) {
    if (!head) throw new Error("the mirror has no such head");
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

  // Every leaf and published body, checked against the newest signed head; a missing body needs a proved withhold.
  async function loadLog(pem) {
    const latest = await get("sth/latest.json");
    if (!latest) throw new Error("the log has no signed head yet");
    let signed = await checkHead(latest, pem);
    const n = latest.tree_size;
    const hashes = ((await get("hashes.json")) || {}).leaves || [];
    if (hashes.length < n) throw new Error("the mirror changed while it was read; try again");
    const leaves = hashes.slice(0, n);
    if (!leaves.every(isHex) || hex(await treeHash(leaves.map(unhex))) !== latest.root_sha256) {
      throw new Error("the mirror's leaf hashes do not make its signed root");
    }
    const bundle = (await get("bodies.json")) || {};
    const texts = bundle.bodies || [];
    const entries = new Array(n).fill(null);
    for (let i = 0; i < n && i < texts.length; i++) {
      if (texts[i] === null || texts[i] === undefined) continue;
      if (hex(await leafHash(enc.encode(texts[i]))) !== leaves[i]) throw new Error("the body of entry " + i + " does not hash to its leaf");
      entries[i] = JSON.parse(texts[i]);
    }
    const withheld = {};
    entries.forEach((e, i) => {
      if (e && e.kind === "withhold" && leaves.indexOf(e.withholds) >= 0 && leaves.indexOf(e.withholds) < i && !withheld[e.withholds]) withheld[e.withholds] = e;
    });
    let pending = bundle.pending_from;
    if (pending !== null && pending !== undefined && pending < n) {
      const first = await get("sth/" + (pending + 1) + ".json");
      signed = (await checkHead(first, pem)) && signed;
      if (hours(first.timestamp, latest.timestamp) > DELAY_ALLOWANCE_DAYS * 24) {
        throw new Error("the mirror holds back bodies from entry " + pending + " on, beyond the publication delay");
      }
    } else {
      pending = null;
    }
    for (let i = 0; i < n; i++) {
      if (!entries[i] && !withheld[leaves[i]] && (pending === null || i < pending)) {
        throw new Error("entry " + i + "'s body is missing and no proved withhold names it");
      }
    }
    const lost = {};
    for (const [leaf, e] of Object.entries(withheld)) if (e.lost === true) lost[leaf] = true;
    return { latest, leaves, entries, withheld, pending, signed, lost };
  }

  function byRecord(h, log) {
    let found = -1;
    log.entries.forEach((e, i) => { if (e && e.kind === "admit" && e.record_sha256 === h) found = i; });
    if (found < 0) throw new Error("no admitted entry for record " + h + (log.pending !== null ? " (some recent bodies are not yet published)" : ""));
    return [found, "the record's admit entry", null, h];
  }

  async function resolve(text, log) {
    const t = text.trim();
    if (isHex(t.toLowerCase())) {
      const h = t.toLowerCase();
      const i = log.leaves.indexOf(h);
      return i >= 0 ? [i, "the leaf hash given", null, null] : byRecord(h, log);
    }
    let doc;
    try { doc = JSON.parse(t); } catch (e) { throw new Error("not JSON and not a sha256 hex digest"); }
    if (doc && doc.version === "guild.register.log.entry/1") {
      const i = log.leaves.indexOf(hex(await leafHash(enc.encode(jcs(doc)))));
      if (i < 0) throw new Error("that entry is not in the log");
      return [i, "the entry pasted, hashed here", doc, null];
    }
    if (doc && isHex(doc.record_sha256)) {
      const r = byRecord(doc.record_sha256, log);
      r[1] = "the record's hash as the pasted document states it; the command also rehashes a full signed record";
      return r;
    }
    throw new Error("neither a log entry nor a register record");
  }

  function strikeOf(log, index) {
    const leaf = log.leaves[index];
    for (let i = index + 1; i < log.leaves.length; i++) {
      const e = log.entries[i];
      if (e && e.kind === "revoke" && e.revokes === leaf) return { reason: e.reason, sitting: e.sitting, withheld: false };
      if (e && e.kind === "withhold" && e.revokes === leaf) return { reason: null, sitting: e.sitting, withheld: true };
    }
    return null;
  }

  // An entry lost in a recovery after this one leaves its strike unknown until a later confirm answers that loss,
  // for every admit (naming none) or for this one.
  function lostAfter(log, index) {
    const leaf = log.leaves[index];
    const answeredAfter = (h, from) => log.entries.slice(from).some((e) => e && e.kind === "confirm" && e.answers === h &&
      (e.confirms === undefined || e.confirms === null || e.confirms === leaf));
    // An admit appended after a recovery waits on the losses it names until a confirm after it answers them.
    for (const h of ((log.entries[index] || {}).pending_lost || [])) if (!answeredAfter(h, index + 1)) return true;
    for (let i = index + 1; i < log.leaves.length; i++) {
      if (!log.lost[log.leaves[i]]) continue;
      const answered = log.entries.slice(i + 1).some((e) => e && e.kind === "confirm" && e.answers === log.leaves[i] &&
        (e.confirms === undefined || e.confirms === null || e.confirms === leaf));
      if (!answered) return true;
    }
    return false;
  }

  function show(lines) {
    out.replaceChildren(...lines.map(([cls, text]) => { const p = document.createElement("p"); if (cls) p.className = cls; p.textContent = text; return p; }));
  }

  async function run(text) {
    show([["", "Checking…"]]);
    const keyDoc = await get("key.json");
    const pem = form.dataset.pinnedPem || (keyDoc && keyDoc.public_key_pem);
    if (!pem) throw new Error("no log key to check against");
    const log = await loadLog(pem);
    const latest = log.latest;
    const [index, how, given, record] = await resolve(text, log);
    const leafHex = log.leaves[index];
    const proof = await get("proofs/" + leafHex + ".json");
    if (!proof || proof.leaf_index !== index) throw new Error("the mirror's inclusion proof is missing or for another position");
    let head = latest;
    if (proof.tree_size !== latest.tree_size) {
      head = await get("sth/" + proof.tree_size + ".json");
      log.signed = (await checkHead(head, pem)) && log.signed;
    }
    await verifyInclusion(unhex(leafHex), index, head.tree_size, proof.audit_path.map(unhex), unhex(head.root_sha256));
    const entry = log.entries[index] || given;
    if (record !== null && !(entry && entry.kind === "admit" && entry.record_sha256 === record)) {
      throw new Error("the entry proved is not an admit of the record asked for");
    }
    const struck = strikeOf(log, index);
    const age = hours(latest.timestamp, new Date().toISOString());
    const what = entry ? entry.kind + " entry" + (entry.serial ? " for No. " + entry.serial : "") : "entry";
    const lines = [["ok", "Proved: the " + what + " is leaf " + index + " of the log at size " + latest.tree_size + "."],
      ["", "Found by " + how + ". Newest head signed " + latest.timestamp + " (" + Math.round(age) + " hours ago) by log key " + latest.key_id + "."]];
    if (age > STALE_HOURS) lines.unshift(["bad", "Not current: the newest head is " + Math.round(age) + " hours old; this copy may be frozen (the log takes a heartbeat entry daily)."]);
    if (!log.signed) lines.push(["bad", "This browser cannot check Ed25519 signatures: the proof holds, but check the head's signature with khaos-log-verify."]);
    if (!form.dataset.pinnedPem) lines.push(["bad", "The key is the one the register serves, not one committed in the contract."]);
    if (struck) lines.push(["bad", "Struck: " + (struck.withheld ? "its reason is withheld" : struck.reason || "") + (struck.sitting ? " (" + struck.sitting + ")" : "")]);
    if (log.withheld[leafHex]) lines.push(["", "The body is withheld, by a withhold entry in the log; the proof stands without it."]);
    else if (!log.entries[index]) lines.push(["", "The body is not yet published in this mirror; the proof stands without it."]);
    let held = 0;
    for (let i = index + 1; i < log.leaves.length; i++) if (!log.entries[i] && !log.withheld[log.leaves[i]]) held++;
    if (held && !struck) lines.push(["bad", "Proved, but newer entries are not yet public; a strike among them would not show (" + held + " held back by the publication delay)."]);
    if (!struck && lostAfter(log, index)) lines.push(["bad", "An entry after this one was lost in a recovery; whether it struck this model is unknown."]);
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
