"""The register's pages in the rooms' own look: the Record sleeve tokens and shapes the Workshop and the Listening Space
draw with (DESIGN.md in the Space is the canon), the Khaos star as the brand, the Guild seal struck on a model's
page from its countersigned record. No webfont is fetched; a locally installed Red Hat Display is used when there.
"""

from __future__ import annotations

import html
import re
from importlib import resources

STAR_SVG = resources.files("khaos_attribution.seal").joinpath("star.svg").read_text(encoding="utf-8")

CSS = """
:root {
  --bg: #070a0b; --panel: #0f1416; --panel-hover: #161d1f; --field: #1c2426;
  --highlight: #2ee5d2; --text: #f0f9f9; --muted: #8a9a9c;
  --fs-note: 12px; --fs-ui: 15px; --fs-lead: 20px; --fs-h2: 28px; --fs-h1: 40px;
  --warn: #f59e0b; --danger: #ff6b6b; --ink-on-highlight: #04211d; --radius: 6px;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html { overflow-y: scroll; scrollbar-gutter: stable; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: 'Red Hat Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  font-size: var(--fs-ui); line-height: 1.55; min-height: 100vh; -webkit-font-smoothing: antialiased;
  font-variant-numeric: tabular-nums;
}
a { color: inherit; }
a:focus-visible { outline: 2px solid var(--highlight); outline-offset: 2px; border-radius: 2px; }
main { max-width: 1080px; margin: 0 auto; padding: 28px 34px 80px; }
.brand { display: flex; align-items: center; gap: 12px; padding: 6px 0 40px; text-decoration: none; }
.brand svg { width: 28px; height: 28px; fill: var(--text); stroke: var(--text); stroke-width: 2px; }
.brand span { font-size: var(--fs-ui); font-weight: 700; }
.brand small { color: var(--muted); font-weight: 600; font-size: var(--fs-note); margin-left: 4px; }
.dev { margin: 0 0 28px; padding: 12px 16px; border-radius: var(--radius);
  background: color-mix(in srgb, var(--warn) 12%, var(--panel)); color: var(--warn); font-size: var(--fs-ui); font-weight: 600; }
.stage-hero { position: relative; margin: 0 0 36px; }
.stage-hero-seal { position: absolute; top: 0; right: 0; z-index: 1; }
.stage-hero-seal svg { display: block; width: 180px; height: 180px; }
.stage-hero:has(.stage-hero-seal) .stage-hero-text { padding-right: 200px; }
.stage-hero-text { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.stage-eyebrow { font-size: var(--fs-note); color: var(--muted); font-weight: 600; }
.stage-title { margin: 0; font-size: var(--fs-h1); font-weight: 900; line-height: 1.02; letter-spacing: -.02em; overflow-wrap: anywhere; }
.stage-para { margin: 0; color: var(--muted); font-size: var(--fs-ui); line-height: 1.6; max-width: 62ch; }
.stage-facts { font-size: var(--fs-note); color: var(--muted); font-weight: 600; }
.stage-section { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin: 0 0 14px; }
.stage-section-title { font-size: var(--fs-h2); font-weight: 700; letter-spacing: -.01em; margin: 0; }
.stage-section-right { font-size: var(--fs-note); color: var(--muted); }
.stage-table { display: grid; grid-template-columns: var(--stage-cols, 1fr); column-gap: 18px; row-gap: 0;
  background: var(--panel); border-radius: var(--radius); padding: 8px; margin-bottom: 32px; }
.stage-thead, .stage-tr { grid-column: 1 / -1; display: grid; grid-template-columns: subgrid; gap: 18px; align-items: center; }
.stage-thead { padding: 8px 12px 10px; font-size: var(--fs-note); color: var(--muted); font-weight: 600; }
.stage-thead > span { min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.stage-tr { padding: 12px; border-radius: var(--radius); font-size: var(--fs-ui); color: var(--text); text-decoration: none; }
.stage-tr:nth-child(even) { background: var(--panel-hover); }
a.stage-tr:hover { background: var(--field); }
.stage-td { min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.stage-td.wrap { white-space: normal; overflow-wrap: anywhere; }
.stage-td.muted, .muted { color: var(--muted); }
.stage-td.num { color: var(--muted); font-weight: 700; }
.stage-kv-label { font-size: var(--fs-note); color: var(--muted); }
.mono { font-size: var(--fs-note); color: var(--muted); overflow-wrap: anywhere; }   /* ids in the text face, tabular */
.stage-chip { display: inline-flex; align-items: center; height: 24px; padding: 0 10px; border-radius: 999px; background: var(--field);
  font-size: var(--fs-note); font-weight: 600; color: var(--muted); white-space: nowrap; }
.stage-chip.good { color: var(--text); }
.stage-chip.failed { color: var(--danger); background: color-mix(in srgb, var(--danger) 14%, var(--panel)); }
.empty { padding: 24px; background: var(--panel); border-radius: var(--radius); color: var(--muted); font-size: var(--fs-ui); }
.foot { margin-top: 48px; padding: 18px 20px; background: var(--panel); border-radius: var(--radius); color: var(--muted);
  font-size: var(--fs-note); display: flex; gap: 18px; flex-wrap: wrap; }
.foot a { color: var(--muted); }
.struck { text-decoration: line-through; text-decoration-color: var(--danger); color: var(--muted); }
.log-form { display: flex; flex-direction: column; gap: 12px; background: var(--panel); border-radius: var(--radius); padding: 16px;
  margin-bottom: 24px; }
.log-form label { font-size: var(--fs-note); color: var(--muted); font-weight: 600; }
.log-form textarea { width: 100%; min-height: 120px; resize: vertical; background: var(--field); color: var(--text); border: 0;
  border-radius: var(--radius); padding: 12px; font: inherit; font-size: var(--fs-note); }
.log-form textarea:focus-visible, .log-form button:focus-visible { outline: 2px solid var(--highlight); outline-offset: 2px; }
.log-form button { align-self: flex-start; height: 40px; padding: 0 18px; border: 0; border-radius: 999px; background: var(--highlight);
  color: var(--ink-on-highlight); font: inherit; font-weight: 700; cursor: pointer; }
.log-result { background: var(--panel); border-radius: var(--radius); padding: 16px; margin-bottom: 24px; }
.log-result:empty { display: none; }
.log-result p { margin: 0 0 8px; }
.log-result .ok { color: var(--highlight); font-weight: 700; }
.log-result .bad { color: var(--danger); font-weight: 700; }
@media (max-width: 760px) {
  main { padding: 20px 16px 60px; }
  .stage-hero:has(.stage-hero-seal) .stage-hero-text { padding-right: 0; padding-top: 150px; }
  .stage-hero-seal svg { width: 130px; height: 130px; }
  .stage-td { white-space: normal; }
  .stage-title { font-size: var(--fs-h2); }
}
"""

DEV_NOTE = "This register signs with a development key. Nothing it countersigns is a Guild record."


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


SITE_NAME = "Guild Register"
DESCRIPTION = ("The public ledger of fine-tuned music models: what each was trained on, as its artist attested, "
               "countersigned by the Guild Register and never edited. Kept by the Khaos Foundation.")
FAVICON_SVG = STAR_SVG.replace("<svg ", '<svg fill="#2ee5d2" stroke="#2ee5d2" stroke-width="6" ', 1)


def head_tags(title: str, description: str, url: str, base: str, extra: str = "") -> str:
    """What search engines and link previews read: the description, the canonical address, the icons and the share card."""
    t, d, u = esc(title), esc(description), esc(url)
    return (f'<meta name="description" content="{d}"><link rel="canonical" href="{u}">'
            f'<link rel="icon" type="image/svg+xml" href="/favicon.svg"><link rel="apple-touch-icon" href="/apple-touch-icon.png">'
            f'<meta name="theme-color" content="#070a0b">'
            f'<meta property="og:type" content="website"><meta property="og:site_name" content="{SITE_NAME}">'
            f'<meta property="og:title" content="{t}"><meta property="og:description" content="{d}"><meta property="og:url" content="{u}">'
            f'<meta property="og:image" content="{esc(base)}/card.png"><meta property="og:image:width" content="1200">'
            f'<meta property="og:image:height" content="630"><meta property="og:image:alt" content="The Guild Register, with the Khaos star">'
            f'<meta name="twitter:card" content="summary_large_image">{extra}')


def page(title: str, body: str, *, dev: bool, description: str = DESCRIPTION, path: str = "/", base: str = "", extra_head: str = "") -> str:
    dev_line = f'<div class="dev"><span>{esc(DEV_NOTE)}</span></div>' if dev else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title>{head_tags(title, description, base + path, base, extra_head)}<style>{CSS}</style></head><body><main>'
            f'<a class="brand" href="/">{STAR_SVG}<span>Guild Register</span><small>of Fine Tuners</small></a>'
            f'{dev_line}{body}'
            f'<div class="foot"><a href="/verify-entry">Verify an entry</a><a href="/log/sth">The log\'s signed head</a>'
            f'<a href="/api/ledger">The ledger</a><a href="/api/public-key">The register\'s public key</a>'
            f'<a href="https://github.com/guild-register/ledger">The public mirror</a><span>Kept by the Khaos Foundation Limited</span></div>'
            f'</main></body></html>')


def site_json_ld(base: str) -> str:
    """The home page as a WebSite kept by the Foundation, for search engines; data, never run."""
    import json  # noqa: PLC0415
    doc = {"@context": "https://schema.org", "@type": "WebSite", "name": "The Guild Register", "url": base + "/",
           "description": DESCRIPTION, "publisher": {"@type": "Organization", "name": "Khaos Foundation Limited"}}
    return '<script type="application/ld+json">' + json.dumps(doc).replace("<", "\\u003c") + "</script>"


def robots_txt(base: str) -> str:
    return f"User-agent: *\nAllow: /\nDisallow: /api/\n\nSitemap: {base}/sitemap.xml\n"


def sitemap_xml(base: str, models: list) -> str:
    """The home page and every model page; ``models`` is (serial, last-changed day) pairs."""
    urls = [f"<url><loc>{esc(base)}/</loc></url>"] + [
        f"<url><loc>{esc(base)}/models/{int(serial)}</loc>" + (f"<lastmod>{esc(day)}</lastmod>" if day else "") + "</url>"
        for serial, day in models]
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            + "".join(urls) + "</urlset>\n")


def hero(eyebrow: str, title: str, para: str, facts: list, seal_svg: str = "") -> str:
    seal = f'<div class="stage-hero-seal">{seal_svg}</div>' if seal_svg else ""
    facts_html = " · ".join(facts)
    brow = f'<div class="stage-eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    return (f'<header class="stage-hero">{seal}<div class="stage-hero-grid"><div class="stage-hero-text">'
            f'{brow}<h1 class="stage-title">{esc(title)}</h1>'
            f'<p class="stage-para">{esc(para)}</p><div class="stage-facts">{facts_html}</div></div></div></header>')


def section(title: str, right: str = "") -> str:
    return f'<div class="stage-section"><h2 class="stage-section-title">{esc(title)}</h2><span class="stage-section-right">{right}</span></div>'


def table(cols: str, headers: list, rows: list) -> str:
    """Rows are lists of cell HTML (already escaped by the caller); a row may be (href, cells) to be a link."""
    head = "".join(f"<span>{esc(h)}</span>" for h in headers)
    body = []
    for row in rows:
        href, cells = (row if isinstance(row, tuple) else (None, row))
        inner = "".join(f'<div class="stage-td{" " + cls if cls else ""}" role="cell">{cell}</div>' for cell, cls in
                        ((c if isinstance(c, tuple) else (c, "")) for c in cells))
        body.append(f'<a class="stage-tr" href="{esc(href)}" role="row">{inner}</a>' if href else f'<div class="stage-tr" role="row">{inner}</div>')
    return (f'<div class="stage-table" role="table" style="--stage-cols: {esc(cols)}"><div class="stage-thead" role="row">{head}</div>'
            + "".join(body) + "</div>")


def state_chip(state: str) -> str:
    kind = {"registered": "good", "withdrawn": "failed", "under review": "failed"}.get(state, "")
    return f'<span class="stage-chip {kind}">{esc(state)}</span>'


def seal_for(serial: int, record: dict) -> str:
    """The Guild seal struck from the countersigned record; empty when the record yields no mark, standard or year."""
    from khaos_attribution.seal import DATE_LETTER_EPOCH, SealRefused, artist_mark_of, render_seal  # noqa: PLC0415
    model = record.get("model") or {}
    mark = record.get("artist_mark") or artist_mark_of(str(record.get("artist_name") or ""))
    parts = str(model.get("card_schema_version") or "").split(".")
    standard = ".".join(parts[:2]) if len(parts) >= 2 else ""
    m = re.match(r"(\d{4})", str(model.get("trained_at_utc") or record.get("published_at_utc") or ""))
    letter = chr(ord("A") + (int(m.group(1)) - DATE_LETTER_EPOCH) % 26) if m else ""
    if not (mark and standard and letter):
        return ""
    try:
        return render_seal(str(serial), str(mark).upper(), standard, letter, size=180)
    except (SealRefused, ValueError):
        return ""


def day(iso) -> str:
    return str(iso or "")[:10]


def count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def log_section(entries: list, struck: dict) -> str:
    """The model's entries in the signed log; an admit struck by a revoke is drawn struck through, with the reason."""
    if not entries:
        return ""
    rows = []
    for e in entries:
        body = e.get("entry") or {}
        kind, lh = e["kind"], e["leaf_hash"]
        strike = struck.get(lh) if kind == "admit" else None
        if kind == "admit":
            what = "Admitted" + (f" at {esc(body['sitting'])}" if body.get("sitting") else "")
        elif kind == "confirm":
            what = "Confirmed after a recovery" + (f" at {esc(body['sitting'])}" if body.get("sitting") else "")
        elif kind == "revoke":
            what = f"Struck: {esc(body.get('reason') or '')}" + (f" ({esc(body['sitting'])})" if body.get("sitting") else "")
        else:
            what = esc(kind)
        if not body:
            what = f"{esc(kind)}, body withheld"
        cell = f'<span class="struck">{what}</span>' if strike else what
        if strike:
            cell += f' <span class="muted">· {esc(strike.get("reason") or "")}</span>'
        rows.append([(cell, "wrap"), (esc(day(body.get("time"))), "muted"),
                     (f'<a class="mono" href="/verify-entry#{esc(lh)}">{esc(lh[:16])}…</a>', "")])
    return section("In the log", "the Guild's signed append-only log") + table(
        "minmax(0, 1fr) 120px 180px", ["Entry", "Date", "Leaf"], rows)


VERIFY_PARA = ("Paste a register record, a log entry, a leaf hash or a record hash. Your browser fetches the log's own files, "
               "recomputes the hashes, walks the inclusion proof to the signed tree head and checks the head's signature. "
               "Nothing is sent anywhere.")


def verify_entry_body(committed, key_doc: dict) -> str:
    """The in-browser verifier's page; the pinned key is the one committed in the contract, when there is one."""
    pinned_pem = committed.public_pem if committed else ""
    if committed and committed.key_id == key_doc["key_id"]:
        key_line = f"Pinned log key {esc(committed.key_id)}, as committed in the contract."
    elif committed:
        key_line = (f"The contract pins log key {esc(committed.key_id)}, but this register serves {esc(key_doc['key_id'])}: "
                    "every check here will fail until they agree.")
    else:
        key_line = (f"No log key is committed in the contract yet; this page checks against the key the register serves "
                    f"({esc(key_doc['key_id'])}{', a development key' if key_doc['development'] else ''}), which proves "
                    "consistency, not who signed.")
    body = hero("Guild Register", "Verify an entry", VERIFY_PARA, [key_line])
    body += (f'<form class="log-form" id="log-form" data-pinned-pem="{esc(pinned_pem)}" data-mirror="/mirror">'
             '<label for="log-input">Record, entry or hash</label>'
             '<textarea id="log-input" spellcheck="false" autocomplete="off"></textarea>'
             '<button type="submit">Verify</button></form>'
             '<div class="log-result" id="log-result" aria-live="polite"></div>'
             '<p class="stage-para">The same check runs offline with <span class="mono">khaos-log-verify</span>, from a copy of '
             'the mirror and the key committed in the contract. A browser without Ed25519 in WebCrypto checks the proof '
             'and says the signature must be checked with the command.</p>'
             '<script src="/log/verify.js" defer></script>')
    return body
