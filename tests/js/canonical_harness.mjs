// Runs the register page's record form under Node: each case is a JSON text; prints one JSON line per case, the form's
// string or {"error": message}.
import fs from "node:fs";
const [, , js, casesPath] = process.argv;
globalThis.document = { getElementById: () => null };
globalThis.__khaosCanonical = {};
new Function(fs.readFileSync(js, "utf8"))();
const { pythonCompact } = globalThis.__khaosCanonical;
for (const text of JSON.parse(fs.readFileSync(casesPath, "utf8"))) {
  try {
    console.log(JSON.stringify(pythonCompact(text)));
  } catch (e) {
    console.log(JSON.stringify({ error: e.message }));
  }
}
