"""The shared engine layer: an engine package reaches another only through engines/*.py, and every name the rooms
import from the engines is still where they import it."""
import ast
from pathlib import Path

import pytest

from khaos_attribution import engines as E

ENGINES = Path(E.__file__).parent
PACKAGES = sorted(p.name for p in ENGINES.iterdir() if (p / "__init__.py").is_file())

# Every (module, name) the Workshop, the Space and the Engine Room import from the engines, read from their sources.
ROOM_IMPORTS = [
    ("khaos_attribution.engines", ["EngineUnavailable", "KNOWN_ENGINES", "capabilities", "descriptor", "engine", "mock",
                                   "provision", "worker_process"]),
    ("khaos_attribution.engines.acestep", ["DESCRIPTOR", "render", "train"]),
    ("khaos_attribution.engines.acestep.batching", ["_CURRENT_MASK", "_FunctionalProxy"]),
    ("khaos_attribution.engines.acestep.heldout", ["DEFAULT_BUCKETS", "HIGH_NOISE_T", "HeldoutEvaluator", "_target_of",
                                                   "bucket_key", "bucket_timesteps", "evaluate_heldout",
                                                   "pair_noise_seed"]),
    ("khaos_attribution.engines.acestep.measure", ["ENGINE_RESIDENT_GB", "ENGINE_YIELDED_GB", "LOW_MEMORY_OVERRIDE_ENV",
                                                   "SCORER_BASE_GB", "SCORER_CHECKPOINTED_GB",
                                                   "SCORER_CHECKPOINTED_GB_PER_SEC", "SCORER_GB_PER_SEC",
                                                   "SCORER_PROCESS_GB", "SYSTEM_RESERVE_GB", "enable_checkpointing",
                                                   "flush_accelerator", "machine_memory_gb", "memory_plan",
                                                   "physical_memory_gb", "scorer_need_gb"]),
    ("khaos_attribution.engines.acestep.render", ["AceStepRenderer", "BASE_MISSING_NOTE", "RenderCancelled",
                                                  "RenderFailed", "RenderRequest"]),
    ("khaos_attribution.engines.acestep.train", ["TrainingUnavailable", "compute_dtype", "dtype_tag",
                                                 "measure_precision", "resolve_device"]),
    ("khaos_attribution.engines.mimetic", ["clip", "measure", "measure_worker", "provision"]),
    ("khaos_attribution.engines.mimetic.loading", ["load_model"]),
    ("khaos_attribution.engines.mimetic.provision", ["venv_python"]),
    ("khaos_attribution.engines.mock", ["train"]),
    ("khaos_attribution.engines.pins", ["GIT_REPO_ENV_OVERRIDES", "checkout_commit", "git_env"]),
    ("khaos_attribution.engines.provision", ["accepted"]),
    ("khaos_attribution.engines.render", ["RenderCancelled", "RenderFailed", "RenderRequest"]),
    ("khaos_attribution.engines.sa3", ["DESCRIPTOR", "loading", "measure", "provision", "render", "train",
                                       "train_driver"]),
    ("khaos_attribution.engines.sa3.heldout_worker", ["heldout_torch", "pair_noise_seed"]),
    ("khaos_attribution.engines.sa3.loading", ["load_model"]),
    ("khaos_attribution.engines.sa3.provision", ["contract_fingerprint"]),
    ("khaos_attribution.engines.sa3.render", ["MLX_SUBDIR", "runtime_for", "venv_python"]),
    ("khaos_attribution.engines.sa3.train", ["DESCRIPTOR", "REPEAT_SMALL_RUNS_ENV", "_env", "pair_files"]),
    ("khaos_attribution.engines.train", ["TrainingUnavailable", "TrainingUpdate", "average_adapters"]),
]


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.lineno, node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name


@pytest.mark.parametrize("package", PACKAGES)
def test_an_engine_package_imports_no_other_engine(package):
    """What two engines share lives in engines/*.py, so setting one engine aside never breaks another."""
    others = {f"khaos_attribution.engines.{p}" for p in PACKAGES if p != package}
    found = [f"{path.name}:{line} imports {module}" for path in sorted((ENGINES / package).glob("*.py"))
             for line, module in _imports(path) if any(module == o or module.startswith(o + ".") for o in others)]
    assert not found, found


def _offered(module: str) -> set:
    """The names a module binds at its top level, read from its source so a module that needs torch is read too."""
    path = ENGINES.parent.joinpath(*module.split(".")[1:])
    source = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
    names = {p.stem for p in path.glob("*.py")} | {p.name for p in path.iterdir() if p.is_dir()} if path.is_dir() else set()
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                names |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
    return names


@pytest.mark.parametrize("module, names", ROOM_IMPORTS, ids=[m for m, _ in ROOM_IMPORTS])
def test_every_name_a_room_imports_from_the_engines_is_still_there(module, names):
    missing = sorted(set(names) - _offered(module))
    assert not missing, f"{module} no longer offers {missing}"


@pytest.mark.parametrize("engine_id", sorted(E.KNOWN_ENGINES))
def test_every_engine_is_its_descriptor_and_three_slots(engine_id):
    """Each package's Engine is the one slots class: its descriptor, and its render, train and measure modules."""
    engine = E.engine(engine_id)
    assert isinstance(engine, E.Engine) and engine.descriptor.id == engine_id
    assert engine.measure.__name__ == E.KNOWN_ENGINES[engine_id] + ".measure"
