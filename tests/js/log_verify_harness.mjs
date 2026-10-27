// Runs the register page's log verifier under Node: each case names a mirror folder, the input and the pinned key;
// prints one JSON line per case with the lines the page would show.
import fs from "node:fs";
const [, , js, casesPath] = process.argv;
const cases = JSON.parse(fs.readFileSync(casesPath, "utf8"));
async function runCase({ mirror, input, pinned }) {
  let shown = null;
  const out = { replaceChildren: (...ps) => { shown = ps.map((p) => (p.className || "") + "|" + p.textContent); } };
  const form = { dataset: { mirror: "/mirror", pinnedPem: pinned }, addEventListener: (ev, fn) => { form.submit = fn; }, requestSubmit() {} };
  globalThis.document = { getElementById: (id) => ({ "log-form": form, "log-result": out, "log-input": { value: input } })[id],
    createElement: () => ({}) };
  globalThis.location = { hash: "" };
  globalThis.fetch = async (url) => {
    const p = mirror + url.replace("/mirror", "");
    if (!fs.existsSync(p)) return { status: 404, ok: false };
    const data = fs.readFileSync(p);
    return { status: 200, ok: true, json: async () => JSON.parse(data.toString()) };
  };
  new Function(fs.readFileSync(js, "utf8"))();
  form.submit({ preventDefault() {} });
  for (let i = 0; i < 400 && (!shown || shown[0].includes("Checking")); i++) await new Promise((r) => setTimeout(r, 5));
  return shown;
}
for (const c of cases) console.log(JSON.stringify(await runCase(c)));
