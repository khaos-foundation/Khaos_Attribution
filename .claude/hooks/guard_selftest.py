"""The guard's self-test: its case table judged by a guard.py, its launcher held to the same answers, its deadline and
its refusal when no python runs. The same file beside every copy of the guard; the caller names the folders it guards.
python3 guard_selftest.py [-v] runs it as the checkout that holds it runs the guard."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
# A row with its own root, its own named folders or the sibling flag is judged as it says, not as a launcher runs.
OWN_ROOT = ("siblings", "root", "repos")
# Every case through the launcher would take minutes: the rows that are not JSON, and every thirtieth of the others.
SPREAD = 30
# The rows Windows reads differently: a backslash kept by quotes divides a path there.
WINDOWS_ROWS = [
    {"bash": 'rm -rf "data\\artists\\x"', "cwd": "{REPO}", "expect": "refused"},
    {"bash": 'cd "data\\artists" && rm -rf x', "cwd": "{REPO}", "expect": "refused"},
    {"bash": "rm -rf 'src\\old'", "cwd": "{REPO}", "expect": "allowed"},
]


def load_guard(path):
    """guard.py run from its source, so no bytecode folder is left beside the hook."""
    module = types.ModuleType("guard")
    with open(path, encoding="utf-8") as source:
        exec(compile(source.read(), "guard.py", "exec"), module.__dict__)
    return module


def load_cases(path):
    with open(path, encoding="utf-8") as table:
        return [case for case in json.load(table) if "group" not in case]


def launched(cases):
    """The rows that also go through the launcher."""
    return [case for case in cases if "raw" in case] + [
        case for case in cases if "raw" not in case and not any(key in case for key in OWN_ROOT)][::SPREAD]


def build(top, name="repo", extra=()):
    """The temp tree the cases name: a repository holding data/, worktrees of it (one linked by a relative path), a
    sibling, a folder that holds artists/ and outputs/, scratch with 300 folders, a repository with no data/, a guarded
    folder that is not a repository, and a folder holding a repository with its own worktree; `extra` under the root."""
    root = os.path.join(top, "root")
    repo = os.path.join(root, name)
    beside = os.path.join(repo, ".claude", "worktrees", "wt")
    worktree = os.path.join(beside, name)
    relative = os.path.join(repo, ".claude", "worktrees", "rel", name)
    holder = os.path.join(top, "holder")
    scratch = os.path.join(top, "scratch")
    ws = os.path.join(top, "ws")
    ws_worktree = os.path.join(ws, ".worktrees", "w")
    folders = [os.path.join(repo, ".git"), os.path.join(repo, "data", "artists", "x"), os.path.join(repo, "src"),
               os.path.join(worktree, "src"), relative, os.path.join(root, "Sibling", ".git"),
               os.path.join(root, "Loose"), os.path.join(beside, "Sibling", ".git"), os.path.join(holder, "artists"),
               os.path.join(holder, "outputs"), scratch, os.path.join(root, "Bare", ".git"),
               os.path.join(root, "Bare", "src"), os.path.join(root, "Notes", "x", "artists"),
               os.path.join(root, "Notes", "run_logs"), os.path.join(ws, ".git"), os.path.join(ws, "Inner", "data"),
               os.path.join(ws, "Inner", "src"), ws_worktree]
    for folder in folders + [os.path.join(root, *path.split("/")) for path in extra]:
        os.makedirs(folder)
    for number in range(300):
        os.makedirs(os.path.join(scratch, "many", "d%03d" % number))
    # git writes the link with forward slashes on Windows too, and may write it relative to the worktree.
    for tree, gitdir in ((worktree, os.path.join(repo, ".git", "worktrees", "wt").replace(os.sep, "/")),
                         (ws_worktree, os.path.join(ws, ".git", "worktrees", "w").replace(os.sep, "/")),
                         (relative, "../../../../.git/worktrees/rel")):
        with open(os.path.join(tree, ".git"), "w", encoding="utf-8") as link:
            link.write("gitdir: " + gitdir + "\n")
    return {"{ROOT}": root, "{NAME}": name, "{REPO}": repo, "{WORKTREE}": worktree, "{HOLDER}": holder,
            "{SCRATCH}": scratch, "{BARE}": os.path.join(root, "Bare"), "{WS}": ws, "{WS_WORKTREE}": ws_worktree,
            "{RELATIVE_WORKTREE}": relative}


def written(path):
    """A path as a shell command carries it: a shell reads a backslash as an escape, so C:\\x is written C:/x."""
    return path.replace("\\", "/")


def git_bash(path):
    """A path in Git Bash's own form: C:\\x is /c/x."""
    return re.sub(r"^([A-Za-z]):", lambda found: "/" + found.group(1).lower(), written(path))


def quoted(path):
    """A Windows path quoted, so the shell keeps its backslashes: "C:\\x". A bare name stays as written."""
    return '"' + path + '"' if os.path.isabs(path) else path


# Git Bash names C:\x as /c/x too, or quoted with its backslashes; those forms are judged only on Windows.
WINDOWS_FORMS = {"git-bash": git_bash, "quoted": quoted}


def fill(text, places, form=str):
    for key, value in places.items():
        text = text.replace(key, form(value))
    return text


def tool_call(case, places, form=written):
    call = {"cwd": fill(case.get("cwd", "{ROOT}"), places)}
    if "bash" in case:
        call.update(tool_name="Bash", tool_input={"command": fill(case["bash"], places, form)})
    else:
        key = "notebook_path" if case["tool"] == "NotebookEdit" else "file_path"
        call.update(tool_name=case["tool"], tool_input={key: fill(case["path"], places)})
    return call


def judge(guard, case, places, root, names=(), form=written):
    """Why the guard refuses the row's call, or None: as the row says, else at `root` with `names` under it."""
    call = tool_call(case, places, form)
    if case.get("siblings"):
        return guard.verdict(call, fill(case.get("root", "{REPO}"), places), (), True)
    if "root" in case or "repos" in case:
        return guard.verdict(call, fill(case.get("root", "{ROOT}"), places), case.get("repos", ()))
    return guard.verdict(call, root, names)


def guard_sh_args(guard_sh):
    """The folders and flags a guard.sh hands guard.py after the root, from its own text."""
    with open(guard_sh, encoding="utf-8") as launcher:
        text = launcher.read().replace("\\\n", " ")
    line = next((each for each in text.splitlines() if '/guard.py" "$ROOT"' in each), None)
    if line is None:
        raise ValueError(guard_sh + " does not hand guard.py the root")
    words = line.split('"$ROOT"', 1)[1].split()
    flags = tuple(word for word in words if word.startswith("--"))
    return tuple(word for word in words if word not in flags), flags


def launcher(guard_sh, stdin, project, path=None):
    """guard.sh's answer to one tool call: allowed (exit 0), refused (exit 2) or the code it ended with."""
    env = dict(os.environ, CLAUDE_PROJECT_DIR=project)
    if path is not None:
        env["PATH"] = path
    done = subprocess.run([shutil.which("bash") or "bash", guard_sh], input=stdin, capture_output=True, text=True,
                          timeout=30, env=env)
    return {0: "allowed", 2: "refused"}.get(done.returncode, "exit %d" % done.returncode), done.stderr.strip()


def launcher_wrong(guard, guard_sh, case, places, project, names, flags):
    """Why guard.sh answers a row otherwise than guard.py would at the root it is given, or None. A row that is not
    JSON has only the table's answer."""
    if "raw" in case:
        got, said = launcher(guard_sh, case["raw"], project)
        want = case["expect"]
    else:
        call = tool_call(case, places)
        got, said = launcher(guard_sh, json.dumps(call), project)
        want = "refused" if guard.verdict(call, project, names, "--siblings-read-only" in flags) else "allowed"
    if got != want or said.startswith("refused: ") != (want == "refused"):
        return "guard.sh said %s (%s), guard.py %s" % (got, said, want)
    return None


DEADLINE = r'''import sys, time, types
guard = types.ModuleType("guard")
exec(compile(open(sys.argv[1], encoding="utf-8").read(), "guard.py", "exec"), guard.__dict__)
guard.LIMIT_SECONDS = 0.3
slow = float(sys.argv[2])
guard.verdict = lambda *a, **k: time.sleep(slow)
sys.argv = ["guard.py", "/nowhere"]
guard.main()
time.sleep(1)
'''


def deadline_wrong(guard_py, slow):
    """Why the deadline misbehaves, or None: a guard still judging at its limit refuses, and one that answered in
    time is never stopped by the timer after. Run in a child, since the refusal ends the process."""
    start = time.monotonic()
    done = subprocess.run([sys.executable, "-I", "-S", "-c", DEADLINE, guard_py, str(slow)],
                          input='{"tool_name": "Bash", "tool_input": {"command": "true"}}', capture_output=True,
                          text=True, timeout=30)
    took = time.monotonic() - start
    code = 2 if slow else 0
    if done.returncode != code or took >= 4 or ("within" in done.stderr) != bool(code):
        return "exit %d after %.1f s: %s" % (done.returncode, took, done.stderr.strip())
    return None


# A python that does not run (Windows' Store stub) or a guard that crashes refuses the call, never allows it.
NO_PYTHON = {"no-python-runs": ("exit 49", "refused: no python"),
             "guard-crashes": ('[ "$3" = -c ] || exit 1', "code 1")}


def no_python_wrong(guard_sh, stub, says, folder):
    """Why guard.sh lets a call through with no python that answers, or None."""
    for name in ("python3", "python", "py"):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as fake:
            fake.write("#!/bin/sh\n%s\n" % stub)
        os.chmod(os.path.join(folder, name), 0o755)
    path = os.pathsep.join([folder, os.path.dirname(shutil.which("bash")), "/usr/bin", "/bin"])
    got, said = launcher(guard_sh, '{"tool_name": "Bash"}', folder, path)
    return None if got == "refused" and says in said else "guard.sh said %s: %s" % (got, said)


def run(hooks, mode, names=(), name="repo", extra=(), with_launcher=True, verbose=False):
    """Every check against the guard in `hooks`, printed; the count of those that went wrong. Mode "repo" judges at
    the repository's root as its guard.sh does; "named" at the root above it with `names` under it."""
    guard_py, guard_sh = os.path.join(hooks, "guard.py"), os.path.join(hooks, "guard.sh")
    guard = load_guard(guard_py)
    cases = load_cases(os.path.join(hooks, "guard_cases.json"))
    wrong = 0
    with tempfile.TemporaryDirectory() as top:
        places = build(os.path.realpath(top), name, extra)
        root = places["{REPO}" if mode == "repo" else "{ROOT}"]
        table = [case for case in cases if "raw" not in case]
        forms = WINDOWS_FORMS.items() if os.name == "nt" else ()
        rows = [(case, written, "") for case in table] + [
            (case, form, " (" + title + ")") for case in table if "bash" in case for title, form in forms]
        rows += [(case, written, " (windows)") for case in (WINDOWS_ROWS if os.name == "nt" else [])]
        for case, form, how in rows:
            why = judge(guard, case, places, root, names, form)
            got = "refused" if why else "allowed"
            if got != case["expect"] or verbose:
                shown = " ".join(case.get("bash", case.get("path", "")).split())[:110]
                miss = "" if got == case["expect"] else " (WRONG, expected %s)" % case["expect"]
                print("%s%s%s  %s  %s" % (got, miss, how, shown, why or ""))
            wrong += got != case["expect"]
        print("%d %s-mode cases, %d wrong" % (len(rows), mode, wrong))
        if not with_launcher:
            return wrong
        if shutil.which("bash") is None:
            print("no bash here to run guard.sh: the launcher checks were NOT run")
            return wrong
        names_sh, flags = guard_sh_args(guard_sh)
        checks = [(case, lambda case=case: launcher_wrong(guard, guard_sh, case, places, root, names_sh, flags))
                  for case in launched(cases)]
        checks += [({"bash": "deadline, " + ("still judging" if slow else "answered")},
                    lambda slow=slow: deadline_wrong(guard_py, slow)) for slow in (5, 0)]
        for title, (stub, says) in NO_PYTHON.items():
            folder = os.path.join(top, title)
            os.makedirs(folder)
            checks.append(({"bash": title}, lambda stub=stub, says=says, folder=folder:
                           no_python_wrong(guard_sh, stub, says, folder)))
        missed = 0
        for case, check in checks:
            why = check()
            if why or verbose:
                shown = case.get("raw", case.get("bash", case.get("path", "")))
                print("launcher %s  %s  %s" % ("WRONG" if why else "ok", " ".join(shown.split())[:110], why or ""))
            missed += bool(why)
        print("%d launcher checks, %d wrong" % (len(checks), missed))
    return wrong + missed


if __name__ == "__main__":
    # A guard.sh that names folders guards them under the root; the table's repository is the first of them.
    names, _ = guard_sh_args(os.path.join(HERE, "guard.sh"))
    sys.exit(1 if run(HERE, "named" if names else "repo", names, names[0] if names else "repo",
                      verbose="-v" in sys.argv) else 0)
