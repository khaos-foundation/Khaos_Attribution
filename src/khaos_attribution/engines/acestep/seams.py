"""The seams Khaos patches in ACE-Step, read from a checkout's source and never imported, so an installer's --check
can run it with no torch: the step normalisation the renderer un-clamps, the codec dtype rule it forces, the DiT's
precision it holds at bf16 on Apple silicon, the base checkpoint's turbo flag, the trainer's precision selector, the
training step's loss, the preprocessor's codec loader and the data module's loader. A source that cannot be read
fails, never passes.
Usage: python -m khaos_attribution.engines.acestep.seams <ACE-Step checkout>"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

HANDLER = "acestep/core/generation/handler"
LORA_MODULE = "acestep/training_v2/fixed_lora_module.py"
PREPROCESS = "acestep/training_v2/preprocess.py"
DATA_MODULE = "acestep/training/data_module.py"


def _parse(root: Path, rel: str, found: list) -> ast.Module | None:
    try:
        return ast.parse((root / rel).read_text(encoding="utf-8"), filename=rel)
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        found.append(f"{rel} cannot be read ({type(exc).__name__}), so its seams cannot be checked")
        return None


def _handler_sources(root: Path, found: list) -> list:
    folder = root / HANDLER
    files = sorted(p for p in folder.glob("*.py") if not p.name.endswith("_test.py")) if folder.is_dir() else []
    if not files:
        found.append(f"{HANDLER} cannot be read, so the handler's seams cannot be checked")
    return [tree for p in files if (tree := _parse(root, p.relative_to(root).as_posix(), found)) is not None]


def _functions(tree: ast.AST, name: str) -> list:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]


def _params(fn) -> set:
    a = fn.args
    return {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}


def _names_in(node: ast.AST) -> set:
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)} | \
        {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
        {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _on_self(tree: ast.AST, name: str, ctx) -> bool:
    return any(isinstance(n, ast.Attribute) and n.attr == name and isinstance(n.value, ast.Name) and n.value.id == "self"
               and isinstance(n.ctx, ctx) for n in ast.walk(tree))


def problems(root) -> list:
    """Every seam the checkout at ``root`` no longer offers as Khaos patches it, in words; empty when all hold."""
    root, found = Path(root), []
    handlers = _handler_sources(root, found)
    # force_mps_bf16 answers the DiT's precision through a property on the class, which only an attribute the
    # handler both sets and reads as self.dtype goes through.
    if handlers and not (any(_on_self(t, "dtype", ast.Store) for t in handlers)
                         and any(_on_self(t, "dtype", ast.Load) for t in handlers)):
        found.append("the handler does not keep the DiT's precision as self.dtype: it cannot be held at bf16 on "
                     "Apple silicon")
    normalise = [f for t in handlers for f in _functions(t, "_normalize_service_generate_inputs")]
    if handlers and not normalise:
        found.append("the handler has no _normalize_service_generate_inputs: the turbo step clamp cannot be lifted")
    elif normalise and not any("infer_steps" in _params(f) and "infer_steps" in _names_in(f) for f in normalise):
        found.append("_normalize_service_generate_inputs takes or returns no infer_steps: the asked steps cannot be restored")
    rule = [f for t in handlers for f in _functions(t, "_get_vae_dtype")]
    callers = [t for t in handlers if any(isinstance(n, ast.Attribute) and n.attr == "_get_vae_dtype"
                                          and isinstance(n.ctx, ast.Load) for n in ast.walk(t))]
    if handlers and (not rule or not callers):
        found.append("the handler's codec calls do not consult _get_vae_dtype: the codec cannot be held at full precision")
    if handlers and not any(_functions(t, "is_turbo_model") for t in handlers):
        found.append("the handler has no is_turbo_model: a turbo checkpoint in the base folder cannot be refused")
    module = _parse(root, LORA_MODULE, found)
    if module is not None:
        selector = _functions(module, "_select_compute_dtype")
        if not selector or "requested_precision" not in _params(selector[0]):
            found.append("_select_compute_dtype takes no requested_precision: the trainer cannot honour a run's precision")
        steps = _functions(module, "training_step")
        if not any("F" in _names_in(s) and "mse_loss" in _names_in(s) for s in steps):
            found.append("training_step computes no F.mse_loss: the masked loss has nothing to mask")
    preprocess = _parse(root, PREPROCESS, found)
    if preprocess is not None:
        at_call = [n for f in ast.walk(preprocess) if isinstance(f, ast.FunctionDef) for n in ast.walk(f)
                   if isinstance(n, ast.ImportFrom) and n.module == "acestep.training_v2.model_loader"
                   and any(a.name == "load_vae" for a in n.names)]
        if not at_call:
            found.append("the preprocessor does not import load_vae when it runs: the codec loader cannot be replaced")
    data = _parse(root, DATA_MODULE, found)
    if data is not None:
        if not _functions(data, "collate_preprocessed_batch"):
            found.append("the data module has no collate_preprocessed_batch: length buckets cannot build their loader")
        modules = [n for n in ast.walk(data) if isinstance(n, ast.ClassDef) and n.name == "PreprocessedDataModule"]
        if not modules or not _functions(modules[0], "train_dataloader"):
            found.append("PreprocessedDataModule has no train_dataloader: the crop and bucket seams have nothing to wrap")
    return found


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__.splitlines()[-1], file=sys.stderr)
        return 2
    found = problems(args[0])
    for line in found:
        print(f"[FAIL] {line}")
    print("CHECK FAILED" if found else "[ok] every seam Khaos patches is where it is patched\nCHECK OK")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
