"""The register's pages in the rooms' own look: the Stage tokens and shapes the Workshop and the Listening Space
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
  --bg: #030809; --panel: #081114; --panel-hover: #0e1a1e; --field: #102226;
  --highlight: #2ee5d2; --highlight-glow: rgba(46, 229, 210, 0.25); --cyan: #80ced7; --text: #f0f9f9;
  --fs-micro: 11px; --fs-note: 12px; --fs-body: 14px; --fs-ui: 15px; --fs-lead: 18px; --fs-h3: 20px; --fs-h2: 26px; --fs-h1: 34px;
  --pill: #15262a; --muted: #80ced7; --border: rgba(128, 206, 215, 0.18); --border-light: rgba(128, 206, 215, 0.3);
  --warn: #f59e0b; --danger: #ff6b6b; --ink-on-highlight: #04211d; --radius: 8px;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html { overflow-y: scroll; scrollbar-gutter: stable; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: 'Red Hat Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  font-size: var(--fs-body); line-height: 1.55; min-height: 100vh; -webkit-font-smoothing: antialiased;
}
a { color: inherit; }
main { max-width: 1080px; margin: 0 auto; padding: 28px 34px 80px; }
.brand { display: flex; align-items: center; gap: 12px; padding: 6px 0 34px; text-decoration: none; }
.brand svg { width: 28px; height: 28px; fill: #fff; stroke: #fff; stroke-width: 2px; }
.brand span { font-size: var(--fs-ui); font-weight: 700; letter-spacing: .02em; }
.brand small { color: var(--muted); font-weight: 600; font-size: var(--fs-note); margin-left: 4px; }
.dev { display: flex; gap: 10px; align-items: baseline; margin: 0 0 26px; padding: 10px 14px; border-radius: 10px;
  background: color-mix(in srgb, var(--warn) 12%, transparent); color: var(--warn); font-size: var(--fs-body); font-weight: 600; }
.stage-hero { position: relative; margin: 0 0 30px; }
.stage-hero-bleed { position: absolute; inset: -60px -34px auto -34px; height: 260px; pointer-events: none;
  background: radial-gradient(70% 90% at 25% 0%, color-mix(in srgb, var(--highlight) 16%, transparent), transparent 70%); }
.stage-hero-grid { position: relative; display: grid; grid-template-columns: minmax(0, 1fr); gap: 40px; align-items: end; }
.stage-hero-seal { position: absolute; top: 0; right: 0; z-index: 1; }
.stage-hero-seal svg { display: block; width: 180px; height: 180px; }
.stage-hero:has(.stage-hero-seal) .stage-hero-text { padding-right: 200px; }
.stage-hero-text { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.stage-eyebrow { font-size: var(--fs-micro); letter-spacing: .1em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.stage-title { margin: 0; font-size: var(--fs-h1); font-weight: 900; line-height: 1.02; letter-spacing: -.02em; overflow-wrap: anywhere; }
.stage-para { margin: 0; color: var(--muted); font-size: var(--fs-ui); line-height: 1.55; max-width: 60ch; }
.stage-facts { font-size: var(--fs-micro); letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.stage-pill-row { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; padding-top: 26px; }
.k-btn { display: inline-flex; align-items: center; justify-content: center; gap: 8px; height: 40px; padding: 0 18px; border-radius: 999px;
  font-family: inherit; font-size: var(--fs-ui); font-weight: 600; line-height: 1; color: var(--text); background: transparent;
  border: 1px solid transparent; cursor: pointer; text-decoration: none; white-space: nowrap; transition: background .12s, color .12s, border-color .15s, box-shadow .15s; }
.k-btn--primary { background: var(--highlight); color: var(--ink-on-highlight); border-color: var(--highlight); }
.k-btn--primary:hover { background: color-mix(in srgb, var(--highlight) 82%, black); border-color: color-mix(in srgb, var(--highlight) 82%, black); box-shadow: 0 0 14px var(--highlight-glow); }
.k-btn--pill { background: var(--pill); color: var(--text); border-color: transparent; }
.k-btn--pill:hover { background: var(--panel-hover); }
.k-btn--secondary { background: transparent; color: var(--highlight); border-color: var(--border-light); }
.k-btn--secondary:hover { background: var(--field); border-color: var(--highlight); }
.stage-section { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; margin: 26px 0 12px; }
.stage-section-title { font-size: var(--fs-h2); font-weight: 700; letter-spacing: -.01em; margin: 0; }
.stage-section-right { font-size: var(--fs-body); color: var(--muted); }
.stage-table { display: grid; grid-template-columns: var(--stage-cols, 1fr); column-gap: 18px; row-gap: 0; }
.stage-thead, .stage-tr { grid-column: 1 / -1; display: grid; grid-template-columns: subgrid; gap: 18px; align-items: center; }
.stage-thead { padding: 0 0 10px; border-bottom: 1px solid var(--border); font-size: var(--fs-micro); letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.stage-thead > span { min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.stage-tr { padding: 12px 0; border-bottom: 1px solid var(--border); font-size: var(--fs-ui); color: var(--text); text-decoration: none; }
.stage-tr:last-child { border-bottom: 0; }
a.stage-tr:hover { background: var(--panel-hover); }
.stage-td { min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.stage-td.wrap { white-space: normal; overflow-wrap: anywhere; }
.stage-td.muted, .muted { color: var(--muted); }
.stage-td.num { color: var(--muted); font-weight: 700; }
.stage-kv-label { font-size: var(--fs-note); color: var(--muted); }
.mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: .85em; color: var(--muted); }
.stage-chip { display: inline-flex; align-items: center; height: 26px; padding: 0 12px; border-radius: 999px; border: 1px solid var(--border-light);
  font-size: var(--fs-note); font-weight: 600; color: var(--muted); white-space: nowrap; }
.stage-chip.good { color: var(--highlight); border-color: var(--highlight); }
.stage-chip.failed { color: var(--danger); border-color: var(--danger); }
.empty { padding: 28px 0; color: var(--muted); font-size: var(--fs-ui); }
.foot { margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--border); color: var(--muted); font-size: var(--fs-note); display: flex; gap: 18px; flex-wrap: wrap; }
.foot a { color: var(--muted); }
@media (max-width: 760px) {
  main { padding: 20px 16px 60px; }
  .stage-hero:has(.stage-hero-seal) .stage-hero-text { padding-right: 0; padding-top: 150px; }
  .stage-hero-seal svg { width: 130px; height: 130px; }
  .stage-td { white-space: normal; }
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
            f'<meta name="theme-color" content="#030809">'
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
            f'<div class="foot"><a href="/api/ledger">The ledger</a><a href="/api/public-key">The register\'s public key</a>'
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
    return (f'<header class="stage-hero"><div class="stage-hero-bleed"></div>{seal}<div class="stage-hero-grid"><div class="stage-hero-text">'
            f'<div class="stage-eyebrow">{esc(eyebrow)}</div><h1 class="stage-title">{esc(title)}</h1>'
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
    kind = {"registered": "good", "withdrawn": "failed"}.get(state, "")
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
