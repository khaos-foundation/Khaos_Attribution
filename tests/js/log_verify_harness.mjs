// Runs the register page's log verifier under Node against a mirror folder; prints one JSON line per case.
import fs from "node:fs";
const [,, js, mirror, infoPath] = process.argv;
const info = JSON.parse(fs.readFileSync(infoPath, "utf8"));
const keyDoc = JSON.parse(fs.readFileSync(mirror + "/log/key.json", "utf8"));
async function runCase(input, pinned, tamper) {
  let shown = null;
  const out = { replaceChildren: (...ps) => { shown = ps.map((p) => (p.className || "") + "|" + p.textContent); } };
  const form = { dataset: { mirror: "/mirror", pinnedPem: pinned }, addEventListener: (ev, fn) => { form.submit = fn; }, requestSubmit() {} };
  const inputEl = { value: input };
  globalThis.document = { getElementById: (id) => ({ "log-form": form, "log-result": out, "log-input": inputEl })[id],
    createElement: () => ({}) };
  globalThis.location = { hash: "" };
  globalThis.fetch = async (url) => {
    const p = mirror + url.replace("/mirror", "");
    if (!fs.existsSync(p)) return { status: 404, ok: false };
    let data = fs.readFileSync(p);
    if (tamper && p.includes("/entries/")) data = Buffer.from(data.toString().replace('"serial":', '"serial":1'));
    return { status: 200, ok: true, json: async () => JSON.parse(data.toString()), arrayBuffer: async () => data.buffer.slice(data.byteOffset, data.byteOffset + data.length) };
  };
  new Function(fs.readFileSync(js, "utf8"))();
  form.submit({ preventDefault() {} });
  for (let i = 0; i < 200 && (!shown || shown[0].includes("Checking")); i++) await new Promise((r) => setTimeout(r, 5));
  return shown;
}
const other = "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAGb9ECWmEzf6FQbrBZ9w7lshQhqowtrbLDFw4rXAxZuE=\n-----END PUBLIC KEY-----\n";
const cases = [[info.records[0], keyDoc.public_key_pem], [info.struck_record, keyDoc.public_key_pem], [JSON.stringify(info.entry), ""],
  [info.leaf, keyDoc.public_key_pem, true], [info.leaf, other]];
for (const c of cases) console.log(JSON.stringify(await runCase(...c)));
