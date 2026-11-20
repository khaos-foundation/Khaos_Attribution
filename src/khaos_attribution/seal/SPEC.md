# The seal and the marks row of the Guild of Fine Tuners — geometry and the constraints

## Naming

The members' body is the **Guild of Fine Tuners** — the Guild, after first
mention. The **Khaos Foundation Limited** holds the Register; its trustees are the
**Guardians**. The Foundation's name is subject to a trade mark question:
it appears in human-readable text only, never in a filename, module name or
public identifier, so a rename is a find-and-replace.

Two assets, one set of punch primitives. Rendered by
`khaos_attribution.seal.render_seal` and `render_marks_row`; bound to a
verified provenance document by `seal_for_provenance` and
`marks_row_for_provenance`.

**The seal** is ceremonial: certificates, the Register page header, anything
physical. The ring inscription — `REGISTER No:` and the register number —
the full interlaced Khaos star presiding, three punches beneath it. The
marks say who examined the model; the ring carries only the number. **The marks row** is small and functional: model cards, model
pages, cover art. Four punches in a line, no ring, no field, a reduced solid
star in the Guild punch.

## Constraint 1: no font, ever

The renderer has **no font dependency and no text element at runtime**: no
`<text>`, no `<textPath>`, no `@font-face`, no network fetch. Text on a
curved path in SVG fails silently across renderers — a browser, Illustrator
and a headless rasteriser each place, kern or drop it differently, and one
of them will do so on the machine that matters. This project has already
lost time to it. Every character is a **vector path** from `glyphs.json`, a
table built once from Red Hat Display Bold by `tools/build_glyphs.py`
(fontTools, build time only; the runtime imports nothing but the standard
library). `tests/test_seal.py` refuses any output containing `<text` or
`<textPath`. Do not reintroduce either; to add a glyph, extend the charset
and rebuild the table.

Charset for inputs: `A-Z 0-9 : . - /` and space. The table also holds one
lowercase glyph, `o`, for the ring label's fixed "No:" — no input may use
it. Each entry: `d` (SVG path, 1000-unit em, y up as the font has it),
`advance`, `bounds` (`[xmin, ymin, xmax, ymax]`, `null` for space).

## Constraint 2: the interlaced star is never small

The interlaced artwork (`star.svg`, from KhaosAI_Star_Dark) appears **only
in the seal**, at 262 units wide. **Never scale it below about 200 units
wide, and never make the reduced form by scaling the full one** — the
interlace does not survive reduction; it fills in and reads as a blot. The
marks row's Guild punch carries a **generated** solid sixteen-point polygon
instead (`reduced_star_points`), and that polygon appears only there. The
tests hold both halves of this.

## The seal

`viewBox="0 0 800 800"`, centre (400, 400); the drawing scales by
`size / 800` through `width`/`height`, never through the viewBox.

| Element | Value |
|---|---|
| Field circle | r 396, fill `#0b0f0e` |
| Outer band rule | r 352, stroke `#ffffff`, width 7, no fill |
| Inner band rule | r 276, stroke `#ffffff`, width 7, no fill |
| Ring inscription | the literal `REGISTER No: ` (capital N, lowercase o, colon, one space) followed by the number; baseline r 290, size 44, tracking 9 after each glyph (none trailing), centred on the top, facing outward |
| Star | `translate(269.00,167.00) scale(0.42429)` — 262 wide, centred (400, 298); fill and stroke `#ffffff`, stroke-width 3, stroke-miterlimit 10 |
| Punch row | `translate(229.80,467.10) scale(0.74)` — three cells 140 × 170, gap 20 (row 460 native), centred (400, 530) |
| Foot ornament | polygon `400,684 409,703 428,712 409,721 400,740 391,721 372,712 391,703`; dots r 8 at (268, 684) and (532, 684) |

### Ring type (exact; implement as given)

**Baseline radius 290 is the approved setting.** 298.6 is the value that
would centre the caps optically between the band rules at 276 and 352; it
was tried and rejected. Do not correct it.

The label is fixed inside the renderer (`RING_PREFIX`), never passed in:
`render_seal` keeps its signature and takes the number alone.

`theta` is 0 at the top and positive clockwise; `n` counts the whole
inscription, label included.

```
adv_i     = glyph advance × s,  s = 44 / 1000       # px
total     = Σ adv_i + 9 × (n − 1)
span      = total / 290
start     = −span / 2                              # centres the run on the top
theta_i   = start + (run_before_i + adv_i / 2) / 290
px        = 400 + 290 · sin(theta_i)
py        = 400 − 290 · cos(theta_i)
glyph:  translate(px, py) rotate(deg(theta_i)) scale(s, −s) translate(−adv_i / (2·s), 0)
```

`scale(s, −s)` flips the font's y-up outlines; the last translate centres
each glyph on its advance; letters face outward as a consequence. A space is
an advance with no mark. Baseline r 290, tallest glyph r ≈ 322, nothing
below r ≈ 289: inside the band (rule inner edges 279.5 and 348.5).

### The number is opaque

The label promises that the number resolves to a Register entry, so it is
an **opaque sequential registration number issued at admission**. This
module never derives it — not from the record hash, the model or anything
else — and never generates, allocates, pads or reformats it: it comes in
from the record and is set exactly as given. It is the Guild Register's
serial for the model (from 65536), which the watermark does not carry: the
watermark names a bucket many models share. No digit count is baked into a
validator. The only refusal is a character the glyph table cannot set.

### Seal punches (within the row group; punch i at x = i × 160)

| i | Punch | Shape | Content | Size / baseline | Rotation | dx, dy |
|---|---|---|---|---|---|---|
| 0 | Artist | canted rect | `artist_mark` | 44 / 101 | −2.4 | 0, 0 |
| 1 | Standard | shield | `standard_version` | 44 / 97 | 1.9 | 1, 3 |
| 2 | Year | cartouche | `date_letter` | 92 / 118 | −1.4 | −1, 2 |

Outlines `fill="none" stroke="#ffffff" stroke-width="9"`. Each punch is
`translate(x + dx, dy) rotate(rotation, 70, 85)` — struck by hand, so each
lands differently.

## The marks row

Four cells 140 × 170, gap 14, padding 34 all round: `viewBox="0 0 670 238"`,
scaled by `height / 238`. Ink `#0b0f0e` on transparent; `reverse=True`
draws white on an ink rectangle.

| i | Punch | Shape | Content | Size / baseline | Rotation | dx, dy |
|---|---|---|---|---|---|---|
| 0 | Artist | canted rect | `artist_mark` | 44 / 101 | −2.4 | 0, 0 |
| 1 | Standard | shield | `standard_version` | 44 / 97 | 1.7 | 1, 3 |
| 2 | Guild | circle r 61 | reduced star | — | −1.2 | −1, −2 |
| 3 | Year | cartouche | `date_letter` | 92 / 118 | 2.9 | 1, 4 |

Outlines stroke-width 7. Cell i sits at x = 34 + i × 154, y = 34.

## Punch primitives (local cell coordinates, 140 × 170, centre (70, 85))

```
canted     M 28,10 L 112,10 L 130,28 L 130,142 L 112,160 L 28,160 L 10,142 L 10,28 Z
shield     M 10,10 L 130,10 L 130,100 C 130,138 104,150 70,161 C 36,150 10,138 10,100 Z
cartouche  M 10,64 C 10,28 36,10 70,10 C 104,10 130,28 130,64 L 130,144 L 112,160 L 28,160 L 10,144 Z
circle     cx 70, cy 85, r 61
reduced star: 16 points, i = 0..15, a = i × 22.5°, r = 50 (even i) or 13 (odd i),
              point = (70 + r·sin a, 85 − r·cos a)
```

Punch type is straight, centred on cx = 70, no tracking, set at the size
given as a **maximum**: a run wider than 100 units is set smaller to fit
the die (a version string may not escape the shield). `artist_mark` is two
or three characters; `date_letter` is one.

## The seal on hero artwork (`seal/artwork.py`)

`place_seal_on_artwork(artwork, number, artist_mark, standard_version,
date_letter, *, colourway=None)` composites the seal onto a model's hero
artwork, **top right**, and returns a **new image** — an SVG document that
embeds the artwork's own bytes untouched and lays the seal over it. The
clean master is never modified: the artist keeps their art without the
mark, and an entry struck from the Register loses its seal without anyone
re-rendering anything. Cache the composite if anything is cached; never
the source. The composite is vector all the way down — no rasterisation
here; a browser or an exporter rasterises at display or export time.

Measured against the artwork's **width** W, whatever its shape:

| Setting | Value |
|---|---|
| Seal diameter | 0.15 W |
| Inset from the top and right edges | 0.05 W, to the seal's bounding box (its unrotated canvas), not its visual edge. `corner="bottom-right"` uses the same inset from the bottom and right edges instead (for exports; the Listening Space's hero shows the sleeve clean, 2026-09-13) |
| Rotation | −4° about the seal's own centre — deliberate; it keeps the seal with the hand-struck punches and stops it reading as a UI badge. Never round it to zero. |

**Colourway.** Two approved references under `assets/`: `guild-seal.svg`
(white artwork on an ink disc, for light artwork) and
`guild-seal-light.svg` (ink artwork on a paper disc `#f4f1ea`, for dark
artwork). The renderer reproduces both (`colourway="ink" | "paper"`), and
`tests/test_seal_artwork.py` holds its geometry to the references to a
thousandth. The choice is made by the **mean luminance of the region the
seal will occupy — never the whole image**; a dark sleeve with a bright
top-right corner is common and the image average gets it wrong. Above
mid-grey the ink disc, below it the paper disc. `colourway=` overrides;
the result records which was used and whether it was sampled or forced.

**Minimum size.** Below 400 px on the output's long edge the register
number in the ring stops being legible and the punches merge, so the
**marks row** is placed instead, bottom right at the same 5 % inset,
0.12 W tall. The row has no paper-ground variant, so its corner rule is
the other way round: a light corner takes the ink marks on the art
itself, a dark corner the white-on-ink row, which carries its own ground.
Below 40 px of row height nothing is placed — the artwork comes back
unmarked and `placed` says `None`. A seal that would not fit the height
(a wide banner) falls back the same way.

## Roughening (off by default)

`rough=True` wraps the outer group in `filter#struck` (fractal noise, base
frequency 0.04, three octaves, seed 11; displacement scale 3.2). Several
renderers ignore filters, so the clean drawing is canonical. The filter must
never carry meaning.

## Provenance binding

`provenance_values(document)` reads the four values from a verified
document — the Listening Space's provenance endpoint's shape: `record` plus
`embedded_agrees_with_sidecar`. It **raises `SealRefused`** unless that flag
is literally `True`. Then, from the record: the number is `register_serial`,
the Guild Register's serial for the model, as given (see "The number is
opaque"); a record without one rings its `watermark_id`, which only a model
from before the Register carries as its own: a derived watermark ID is a
bucket many models share and names none of them. `model_seal_document(card,
serial=, verified=, derived=)` builds a model's document from its card for
every room, and refuses a derived bucket that has no serial yet;
`artist_mark`, `standard_version` and `date_letter` are read from those
keys when the record carries them, else derived from what it does carry —
the initials of `artist_name`, the `schema_version` as major.minor, the
year letter of `timestamp` (2026 → A). Every reading is a function of the
record; nothing is generated, assigned, looked up or revoked here.

`jitter_from_record=True` (default off, not used in any golden, not yet a
decision) draws each punch's rotation and offset from the record's SHA-256
within the design's ranges (rotation ±3°, dx −1…1, dy −2…4), so each
model's marks are subtly and reproducibly its own.

Numbers are formatted to three fixed decimals, so the same input is the
same bytes on every machine.

## Licence

Red Hat Display is © Red Hat, Inc., SIL Open Font License 1.1, Reserved
Font Name "Red Hat". `glyphs.json` is a derived work: permitted under the
OFL provided the licence travels with it (`OFL.txt`, beside this file) and
the derived file carries no reserved name. The TTF itself is not committed.
