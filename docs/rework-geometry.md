# Rework geometry — the cross-app record contract

Both apps let a listener regenerate part of a piece (repaint) or lengthen
it (extend). Both draw the piece's cumulative rework history as tinted
spans on a waveform strip, by walking the chain of source references. The
two implementations are separate by design (no shared JS), so this page is
the single written contract they both implement. A change to one side
without the other is a contract break, not a style choice.

## The geometry every rework output must record

| field | meaning |
|---|---|
| `repaint_start` / `repaint_end` | the regenerated window, in seconds of the OUTPUT's own timeline |
| `extend_head` | seconds prepended by a start-extend (absent/None otherwise) |
| tail length | Workshop records `extend_tail` + measured `extend_from`; the Platform derives it as `repaint_end − repaint_start` when `extend_head` is absent |
| `repaint_regions` | *(since 0.48.0)* every part of a fix that masks several parts in one pass, `[[start, end], ...]` in seconds of the output's timeline; `repaint_start` / `repaint_end` then hold their span, so a reader that knows one window still bounds them |

Rules that keep the records honest:

- **Explicit beats inferred.** Head-ness is a recorded field, never deduced
  from the window's shape. (A start-extend's window happens to begin at 0;
  a repaint window can too. The Platform's outputs API is the one sanctioned
  place that falls back to the `start == 0` inference, and only for records
  written before `extend_head` was persisted.)
- **Measured beats requested.** Chained extends must record the source's
  MEASURED length (Workshop: `extend_from`), not the form value — a stale
  requested duration once plotted a parent's tail as the child's new audio.
- **Several parts, one pass, where the engine can.** Stable Audio 3 masks
  every saved part in one render and records each in `repaint_regions`; an
  engine with one window (ACE-Step) is still driven a part at a time.
- **One side per pass.** A repaint window is one contiguous range, so an
  extend grows the head or the tail, never both in one generation. (The
  Platform's rework panel lets a listener ask for both ends at once; it
  queues two generations — tail first, then the head on the result — so
  every record still carries one window.)
- An engine `repaint_end` of `-1`/None means the SOURCE's end — it appends
  nothing. A tail-extend must record and send an explicit end past the
  source's length.

## The chain walk both strips implement

Walk `ref_audio_job` (Workshop) / `source_generation_id` (Platform) upward
until the first ancestor that is neither a repaint nor an extend — that is a
full regeneration; nothing above it survives into the audio. While walking:

- Own span bright, ancestors dim, oldest painted first.
- `curDur` bounds each ancestor: a tail-extend's parent ended at the
  window's start; a repaint preserves length.
- **Head offsets accumulate upward**: ascending past a head-extend shifts
  every OLDER span rightward by `extend_head` and shortens the parent by
  the same amount. All spans plot at `recorded + accumulated offset`.

## Where each side implements this

- Workshop: `scripts/infer.py` (writes), `src/webapp/static/js` inference
  feed (walks).
- Platform: `engine/acestep.py` + `engine/mock.py` (write), `routers/
  outputs.py` (ships `repaint: {start, end, head}`), `static/space/
  05-families.js` (walks).
- Stable Audio 3, in this contract: `engines/sa3/rework.py` plans the parts on
  the latent grid and splices them into the source; `engines/sa3/render.py`
  records `regions`, `noise_level`, `crossfade_seconds` and an extend's
  `chunks` in the render's params.
