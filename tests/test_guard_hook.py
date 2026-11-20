"""The guard (.claude/hooks/guard.py and guard.sh), by the self-test beside it (guard_selftest.py, guard_cases.json):
every row of the case table, the launcher's answers, the deadline, and the refusal when no python runs."""

import os
import shutil
import types
from pathlib import Path

import pytest


def _hooks():
    """The .claude/hooks of the repository that holds this file, wherever under it the file sits."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".claude" / "hooks" / "guard_selftest.py").is_file():
            return parent / ".claude" / "hooks"
    raise RuntimeError("no .claude/hooks/guard_selftest.py above " + __file__)


HOOKS = _hooks()
GUARD_PY, GUARD_SH = str(HOOKS / "guard.py"), str(HOOKS / "guard.sh")
# Run from its source, so no bytecode folder is left beside the hook.
SELFTEST = types.ModuleType("guard_selftest")
SELFTEST.__file__ = str(HOOKS / "guard_selftest.py")
exec(compile((HOOKS / "guard_selftest.py").read_text(encoding="utf-8"), "guard_selftest.py", "exec"), SELFTEST.__dict__)
CASES = SELFTEST.load_cases(HOOKS / "guard_cases.json")
TABLE = [case for case in CASES if "raw" not in case]
# A guard.sh that names folders guards them under the root above; one that names none guards this repository.
NAMES, FLAGS = SELFTEST.guard_sh_args(GUARD_SH)
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="no bash here to run the hook")


def _named(case):
    return " ".join(str(case.get("raw", case.get("bash", case.get("path")))).split())[:60]


@pytest.fixture(scope="module")
def guard():
    return SELFTEST.load_guard(GUARD_PY)


@pytest.fixture(scope="module")
def places(tmp_path_factory):
    return SELFTEST.build(str(tmp_path_factory.mktemp("guard").resolve()), NAMES[0] if NAMES else "repo")


def _root(places):
    return places["{ROOT}" if NAMES else "{REPO}"]


@pytest.mark.parametrize("case", TABLE, ids=_named)
def test_a_tool_call_is_refused_or_allowed_as_the_table_says(case, guard, places):
    """Each row is one Edit, Write or shell command; a row with its own root or flag is judged as it says."""
    why = SELFTEST.judge(guard, case, places, _root(places), NAMES)
    assert ("refused" if why else "allowed") == case["expect"], why


# Defined only on Windows, where these forms exist: elsewhere an empty set would be a skip some suites refuse.
if os.name == "nt":
    @pytest.mark.parametrize("form", list(SELFTEST.WINDOWS_FORMS))
    @pytest.mark.parametrize("case", [case for case in TABLE if "bash" in case], ids=_named)
    def test_a_windows_path_in_any_form_a_shell_takes_is_judged_as_the_table_says(case, form, guard, places):
        """A shell command may name C:\\x as /c/x or as "C:\\x"; the guard judges the folder Windows opens."""
        why = SELFTEST.judge(guard, case, places, _root(places), NAMES, SELFTEST.WINDOWS_FORMS[form])
        assert ("refused" if why else "allowed") == case["expect"], why

    @pytest.mark.parametrize("case", SELFTEST.WINDOWS_ROWS, ids=_named)
    def test_a_quoted_backslash_divides_a_path_on_windows(case, guard, places):
        why = SELFTEST.judge(guard, case, places, _root(places), NAMES)
        assert ("refused" if why else "allowed") == case["expect"], why


@needs_bash
@pytest.mark.parametrize("case", SELFTEST.launched(CASES), ids=_named)
def test_the_launcher_gives_the_guards_answer(case, guard, places):
    """guard.sh is what the settings run: it hands the call to guard.py and carries its answer in the exit code."""
    assert SELFTEST.launcher_wrong(guard, GUARD_SH, case, places, _root(places), NAMES, FLAGS) is None


@needs_bash
@pytest.mark.parametrize("title", list(SELFTEST.NO_PYTHON))
def test_the_launcher_refuses_when_no_python_answers(title, tmp_path):
    """A python that does not run (Windows' Store stub) or a guard that crashes refuses the call, never allows it."""
    stub, says = SELFTEST.NO_PYTHON[title]
    assert SELFTEST.no_python_wrong(GUARD_SH, stub, says, str(tmp_path)) is None


@pytest.mark.parametrize("slow", [5, 0], ids=["still-judging", "answered"])
def test_a_guard_still_judging_at_its_limit_refuses_and_one_that_answered_is_left_alone(slow):
    """Claude Code lets a call through when the hook outlasts its timeout, so the guard refuses first."""
    assert SELFTEST.deadline_wrong(GUARD_PY, slow) is None
