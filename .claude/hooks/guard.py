"""PreToolUse guard: nothing under a folder named data, artists or outputs is edited or removed from a session.
Called by guard.sh with the tool call on stdin, then the root and the repositories under it (none: the root is one).
"""
import json
import os
import re
import shlex
import sys

PROTECTED = ("data", "artists", "outputs")
# Tracked notes inside a protected folder; they are source and stay editable.
TRACKED_NOTES = ("artists/README.md",)
REMOVERS = ("rm", "rmdir", "unlink", "shred", "truncate", "grm", "grmdir", "gunlink", "gshred", "gtruncate")
FINDERS = ("find", "gfind")
SHELLS = ("bash", "sh", "zsh", "dash", "ksh")
WRAPPERS = ("sudo", "env", "command", "exec", "nice", "nohup", "time", "timeout", "caffeinate", "xargs")
KNOWN = REMOVERS + FINDERS + SHELLS + WRAPPERS + ("rsync", "git", "eval", "trap", "ssh")
# Commands whose words are only text, or are only read, copied or moved: what they are handed is never judged.
HARMLESS = ("echo", "printf", "cat", "grep", "egrep", "fgrep", "rg", "sed", "ls", "gh", "head", "tail", "wc", "diff",
            "sort", "uniq", "cut", "tr", "cp", "mv", "mkdir", "test", "[", "[[")
# Commands that never run what a pipe hands them, though their own words may not be harmless.
READERS = HARMLESS + ("git", "awk", "tee", "jq")
KEYWORDS = ("if", "then", "elif", "else", "do", "while", "until", "!", "{", "export", "local", "declare", "readonly")
GIT_VALUE_OPTIONS = ("-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--config-env")
RSYNC_VALUE_OPTIONS = ("--exclude", "--include", "--filter", "-f", "-e", "--rsh", "--files-from", "--exclude-from",
                       "--include-from")
FIND_PATTERNS = ("-path", "-ipath", "-wholename", "-name", "-iname", "-regex")
SSH_VALUE_OPTIONS = "bcDEeFIiJLlmOopQRSWw"
ASSIGNMENT = re.compile(r"[A-Za-z_]\w*\+?=")
VARIABLE = re.compile(r"\$(\w+)|\$\{(\w+)\}")
REDIRECT = re.compile(r"\d*(?:<|>>?|&>>?)(.*)")
GLOB = re.compile(r"[*?\[]")
HEREDOC = re.compile(r"<<(-?)[ \t]*((?:\\.|'[^']*'|\"[^\"]*\"|[^\s;&|()<>'\"\\])+)")
# A heredoc is a script when a shell is in its pipeline, or when it is written to a shell script's file.
SHELL_WORD = re.compile(r"(?<![\w.-])(?:bash|sh|zsh|dash|ksh|ssh)(?![\w.-])|>\s*[^\s;|&<>]*\.(?:sh|bash|zsh)(?![\w.])")
INTERPRETER = r"(?:python[\d.]*|perl|ruby|node|osascript)"
INTERPRETER_WORD = re.compile(r"(?<![\w.-])" + INTERPRETER + r"(?![\w.-])")
# Code cannot be read as shell: a call that removes, and a protected folder written as a file path (not a URL).
CODE_REMOVE = re.compile(r"rmtree|removedirs|os\.(?:remove|unlink|rmdir)|\.(?:unlink|rmdir)\(|(?:rm|unlink|rmdir)Sync"
                         r"|[\"']rm[\"']|(?<![\w.-])rm\s+-|(?<![\w.])unlink\b")
CODE_PATH = re.compile(r"(?<![A-Za-z0-9_])(?<!api/)(?<!operator/)(?:data|artists|outputs)/")
CODE_REMOVE_NAMED = re.compile(r"(?:rmtree|removedirs|remove|unlink|rmdir|rmSync|unlinkSync|rmdirSync)\(\s*[rfb]?[\"']"
                               r"[^\"']*(?<![A-Za-z0-9_])(?:data|artists|outputs)[\"']")
CRUDE_VERB = re.compile(r"(?<![A-Za-z0-9_])(?:rm|remove|unlink|shred|truncate)|-delete|--del")
# In the words of a command the guard does not model, a removing word is not an option (--rm) or part of a path.
HANDED_VERB = re.compile(r"(?<![\w./-])(?:rm|remove|unlink|shred|truncate)|(?<![\w-])-delete|--del")
CRUDE_PATH = re.compile(r"(?<![A-Za-z0-9_])(?:data|artists|outputs)/")
# The removing word of the floor rule: a line that holds one the guard cannot place, and a protected path, is refused.
REMOVING_WORD = re.compile(r"(?<![A-Za-z0-9_-])rm(?:dir|tree|Sync|dirSync)?(?![A-Za-z0-9_])")
# Where a loop or a group ends: what it printed may be piped on from there.
CLOSERS = ("done", "fi", "esac", "}")


class Refuse(Exception):
    pass


def protected_under(full, bases, notes=()):
    """The protected folder a path sits under inside one of the repositories, or None."""
    for base in bases:
        if not full.startswith(base + os.sep):
            continue
        parts = os.path.relpath(full, base).replace(os.sep, "/").split("/")
        for i, part in enumerate(parts[:-1]):
            if part.lower() in PROTECTED:
                if "/".join(parts[i:]) in notes:
                    break
                return part
    return None


# --- Edit, Write, NotebookEdit ---

def judge_edit(path, cwd, bases):
    given = os.path.abspath(os.path.join(cwd, path))
    # A protected folder may be a link to another disk, so the path is judged as written and as resolved.
    for full in {given, os.path.realpath(given)}:
        name = protected_under(full, bases, TRACKED_NOTES)
        if name:
            raise Refuse(path + " is under a folder named " + name + "; artist material is never edited from a session")


def judge_sibling(path, cwd, root):
    """Refuse a write into another repository: a folder beside this one, or beside its main checkout, holding .git."""
    given = os.path.abspath(os.path.join(cwd, path))
    parents, own = set(), {os.path.basename(root)}
    gitfile = os.path.join(root, ".git")
    if os.path.isfile(gitfile):
        gitdir = open(gitfile, encoding="utf-8").read().split("gitdir:", 1)[-1].strip()
        main_checkout = gitdir.split(os.sep + ".git" + os.sep + "worktrees")[0]
        if main_checkout != gitdir:
            parents.add(os.path.dirname(main_checkout))
            own.add(os.path.basename(main_checkout))
    parents.add(os.path.dirname(root))
    for full in {given, os.path.realpath(given)}:
        for parent in parents | {os.path.realpath(p) for p in parents}:
            if full.startswith(parent + os.sep):
                first = os.path.relpath(full, parent).split(os.sep)[0]
                if first not in own and os.path.exists(os.path.join(parent, first, ".git")):
                    raise Refuse(path + " is inside the " + first + " repository; this one never writes to another")


# --- Bash: the line is split into simple commands, and only a removing command's own words are read ---

def closing(text, i, quote, escapes=True):
    """Index of the quote that closes the one at text[i]."""
    j = i + 1
    while j < len(text):
        if escapes and text[j] == "\\":
            j += 2
        elif text[j] == quote:
            return j
        else:
            j += 1
    raise ValueError("no closing quote")


def substitution(text, i, nested):
    """Where a command substitution opens at text[i]: queue its body and return the index after it; else None."""
    if text[i] == "`":
        j = closing(text, i, "`")
        nested.append(text[i + 1:j].replace("\\`", "`"))
        return j + 1
    if text.startswith("$(", i):
        _, inside, j = scan(text, i + 2, True)
        # $(( )) is arithmetic, not a command: only the substitutions inside it are commands.
        nested.extend(inside if text.startswith("$((", i) else [text[i + 2:j]])
        return j + 1
    return None


def double_quoted(text, i, nested):
    """The double-quoted string opening at text[i], each substitution replaced by $_, and the index after it."""
    out, j = ['"'], i + 1
    while j < len(text):
        if text[j] == "\\":
            out.append(text[j:j + 2])
            j += 2
        elif text[j] == '"':
            return "".join(out) + '"', j + 1
        else:
            k = substitution(text, j, nested)
            out.append(text[j] if k is None else "$_")
            j = j + 1 if k is None else k
    raise ValueError("no closing quote")


def code_removes(code):
    return bool(CODE_REMOVE.search(code) and CODE_PATH.search(code) or CODE_REMOVE_NAMED.search(code))


def occurrences(text):
    return len(REMOVING_WORD.findall(text))


def heredoc_bodies(cmd, i, pending, nested, pieces):
    """Skip the heredoc bodies starting at cmd[i]. One that is run is queued; one that is only text is not read,
    and the removing words in it are counted for the command it is handed to."""
    for delimiter, strip_tabs, quoted, pipeline, owner in pending:
        line, body, ended = cmd[pipeline:i], [], False
        while i < len(cmd) and not ended:
            j = cmd.find("\n", i)
            j = len(cmd) if j < 0 else j
            text, i = cmd[i:j], j + 1
            ended = (text.lstrip("\t") if strip_tabs else text) == delimiter
            if not ended:
                body.append(text)
        body = "\n".join(body)
        # A body with no end may not be a heredoc at all (a << inside arithmetic), so it is read as commands too.
        if SHELL_WORD.search(line) or not ended:
            nested.append(body)
        elif INTERPRETER_WORD.search(line):
            if code_removes(body):
                raise Refuse("the code in the heredoc holds a removing call and a data/, artists/ or outputs/ path")
        elif quoted:
            pieces[owner][3] += occurrences(body)
        else:
            pieces[owner][3] += occurrences(double_quoted('"' + body.replace('"', " ") + '"', 0, nested)[0])
    return min(i, len(cmd))


def scan(cmd, i=0, bracketed=False):
    """Split a command line into simple commands: [text, a pipe feeds it, removing words in its comments, and in the
    heredoc text it is handed]; also the command lines nested in it, and where it ended. With `bracketed`, the line
    is the inside of a $( ) and ends at its bracket."""
    pieces, nested, buf, pending, pipeline, depth, piped, comments = [], [], [], [], i, 0, False, 0
    while i < len(cmd):
        c, after = cmd[i], cmd[i + 1:i + 2]
        last = buf[-1][-1] if buf else " "
        if c == "\\":
            if after != "\n":
                buf.append(cmd[i:i + 2])
            i += 2
        elif c == "'":
            j = closing(cmd, i, c, escapes=False)
            buf.append(cmd[i:j + 1])
            i = j + 1
        elif c == '"':
            text, i = double_quoted(cmd, i, nested)
            buf.append(text)
        elif c == "$" and after == "'":
            j = closing(cmd, i + 1, "'")
            buf.append(shlex.quote(re.sub(r"\\(.)", r"\1", cmd[i + 2:j])))
            i = j + 1
        elif c == "$" and after == '"':
            i += 1
        elif c == "#" and last.isspace():
            j = cmd.find("\n", i)
            j = len(cmd) if j < 0 else j
            comments += occurrences(cmd[i:j])
            i = j
        elif cmd.startswith("<<", i) and cmd[i + 2:i + 3] != "<" and last != "<":
            found = HEREDOC.match(cmd, i)
            if not found:
                raise ValueError("heredoc without a delimiter")
            word = re.sub(r"[\\'\"]", "", found.group(2))
            pending.append((word, bool(found.group(1)), word != found.group(2), pipeline, len(pieces)))
            buf.append(" ")
            i = found.end()
        elif c == "&" and (after == ">" or last in "<>"):
            buf.append(c)
            i += 1
        elif c == "|" and last == ">":
            buf.append(c)
            i += 1
        elif c in ";&|()\n":
            pieces.append(["".join(buf), piped, comments, 0])
            if c == ")" and bracketed and depth == 0:
                return pieces, nested, i
            depth += (c == "(") - (c == ")")
            operator = cmd[i:i + 2] if cmd[i:i + 2] in ("&&", "||", "|&", ";;") else c
            buf, piped, comments = [], operator in ("|", "|&"), 0
            i += len(operator)
            if c == "\n" and pending:
                i = heredoc_bodies(cmd, i, pending, nested, pieces)
                pending = []
            if not piped:
                pipeline = i
        else:
            k = substitution(cmd, i, nested)
            buf.append(c if k is None else "$_")
            i = i + 1 if k is None else k
    if bracketed:
        raise ValueError("no closing bracket")
    return pieces + [["".join(buf), piped, comments, 0]], nested, i


def named(word):
    """True when a word holds a protected folder name as a path segment."""
    # A word with a space in it may be a sentence, so only its slashes divide it.
    dividers = r"/" if re.search(r"\s", word) else r"[/:={},]"
    return any(part.lower() in PROTECTED for part in re.split(dividers, word))


def reaches(word, cwd):
    """True when a word names a protected folder, or holds a glob that expands to one."""
    if named(word):
        return True
    if "$" in word or not GLOB.search(word):
        return False
    import fnmatch
    import glob
    parts = word.split("/")
    for i, part in enumerate(parts):
        for name in PROTECTED:
            if GLOB.search(part) and fnmatch.fnmatch(name, part.lower()):
                if cwd is None:
                    return True
                stem = os.path.expanduser("/".join(parts[:i] + [name]))
                if next(glob.iglob(os.path.join(glob.escape(cwd), stem)), None) is not None:
                    return True
    return False


def expand(word, variables):
    """The word with each variable the line itself sets replaced by its values; any other is left as written."""
    out = [word]
    for found in VARIABLE.finditer(word):
        values = variables.get(found.group(1) or found.group(2))
        if values:
            out = [text.replace(found.group(0), value) for text in out for value in values][:64]
    return out


def removing(what, args, state, fed, feed, globs=True):
    """Refuse a removing command when its own words, or the words that feed it its paths, reach a protected folder."""
    if state["inside"]:
        raise Refuse(what + " would run inside a data, artists or outputs folder")
    unread = False
    for arg in args:
        for text in expand(arg, state["vars"]):
            if reaches(text, state["cwd"]) if globs else named(text):
                raise Refuse(what + " names " + text)
            unread = unread or "$" in text
    for word in feed if fed else []:
        for text in expand(word, state["vars"]):
            if named(text):
                raise Refuse(what + " is fed its paths by a command that names " + text)
    # A path in a variable or substitution the guard cannot follow: the crude rule, a protected path anywhere.
    if unread and CRUDE_PATH.search(state["text"]):
        raise Refuse(what + " takes a path that is not written out, and the line holds a data/, artists/ or "
                     "outputs/ path")


def change_folder(args, state):
    """Follow a cd: its target is its first word that is neither an option nor a redirect."""
    targets, skip = [], False
    for arg in args:
        redirect = REDIRECT.fullmatch(arg)
        if redirect or skip:
            skip = bool(redirect) and not redirect.group(1)
        elif arg == "-" or not arg.startswith("-"):
            targets.append(arg)
    reached = expand(targets[0], state["vars"]) if targets else ["~"]
    target = reached[0]
    # A protected name counts in any value the target may take, even where the rest of it cannot be followed.
    if any(named(text) for text in reached):
        state["inside"] = True
    if "$" in target or target == "-" or len(reached) > 1:
        state["cwd"] = None
        return
    target = os.path.expanduser(target)
    if os.path.isabs(target) and not named(target):
        state["inside"] = False
    if os.path.isabs(target) or state["cwd"] is not None:
        state["cwd"] = os.path.normpath(os.path.join(state["cwd"] or "/", target))


def git(args, state, fed, feed):
    i, before = 0, []
    while i < len(args) and args[i].startswith("-"):
        step = 2 if args[i] in GIT_VALUE_OPTIONS else 1
        before += args[i:i + step]
        i += step
    if args[i:i + 1] != ["clean"]:
        return
    rest = args[i + 1:]
    flags = rest[:rest.index("--")] if "--" in rest else rest
    if any(re.match(r"-[^-]", flag) and re.search(r"[xX]", flag) for flag in flags):
        raise Refuse("git clean with -x or -X removes every ignored folder")
    removing("git clean", before + rest, state, fed, feed)


def find_targets(args):
    """Where a find looks: its starting points, and the patterns it neither negates nor prunes."""
    targets, i = [], 0
    while i < len(args) and args[i] not in ("(", "!") and not re.match(r"-..", args[i]):
        if not args[i].startswith("-"):
            targets.append(args[i])
        i += 1
    for j in range(i, len(args) - 1):
        negated = j > 0 and args[j - 1] in ("-not", "!")
        if args[j] in FIND_PATTERNS and not negated and args[j + 2:j + 3] != ["-prune"]:
            targets.append(args[j + 1])
    return targets


def find_starts(args):
    """The folders a find starts from: its words before the first test or option, else the folder it runs in."""
    starts = []
    for arg in args:
        if arg in ("(", "!") or re.match(r"-..", arg):
            break
        if not arg.startswith("-"):
            starts.append(arg)
    return starts or ["."]


def find_depth_first(args):
    """True when a find works depth first (-delete implies it), where -prune does nothing."""
    return any(arg in ("-delete", "-depth", "-d") for arg in args)


def find_narrowed(args):
    """True when a find picks what it acts on by name or path: a test neither negated nor a prune's own."""
    return any(arg in FIND_PATTERNS and not (j > 0 and args[j - 1] in ("-not", "!"))
               and args[j + 2:j + 3] != ["-prune"] for j, arg in enumerate(args))


def find_leaves_out(args, start, held):
    """True when a find prunes the protected folder `held` under start. A negated test spares only the folder's own
    entry, not what is under it, so it does not count; nor does a prune under depth-first order."""
    import fnmatch
    if find_depth_first(args):
        return False
    start = start.rstrip("/") or "/"
    for j in range(len(args) - 1):
        test, pattern = args[j], args[j + 1]
        if test not in FIND_PATTERNS or test == "-regex" or args[j + 2:j + 3] != ["-prune"]:
            continue
        fold = str.lower if test.startswith("-i") else str
        names = [held] if test in ("-name", "-iname") else [start + "/" + held, held]
        if any(fnmatch.fnmatchcase(fold(name), fold(pattern)) for name in names):
            return True
    return False


def runs_remover(words):
    """True when a command line, through any wrappers before it, is a removing command."""
    cmd = bare(words)
    while cmd and os.path.basename(cmd[0]) in WRAPPERS:
        i = recognised(cmd[1:])
        if i is None:
            return False
        cmd = cmd[1 + i:]
    return bool(cmd) and os.path.basename(cmd[0]) in REMOVERS


def rsync_leaves_out(args, held):
    """True when an rsync excludes the protected folder `held` at the top of what it copies into. With
    --delete-excluded what is excluded is removed, so nothing is left out."""
    import fnmatch
    if any(arg.startswith("--delete-excluded") for arg in args):
        return False
    for i, arg in enumerate(args):
        pattern = None
        if arg.startswith("--exclude="):
            pattern = arg.split("=", 1)[1]
        elif arg == "--exclude" and i + 1 < len(args):
            pattern = args[i + 1]
        elif arg in ("-f", "--filter") and i + 1 < len(args) and args[i + 1][:2] in ("- ", "-/"):
            pattern = args[i + 1][2:]
        if pattern is not None and fnmatch.fnmatchcase(held, re.sub(r"/\**$", "", pattern.strip()).lstrip("/")):
            return True
    return False


def too_wide(what, targets, state, leaves_out=None, root_ok=False):
    """Refuse a removal whose target is a repository's root, or a folder that directly holds a protected folder that
    exists, unless `leaves_out(target, name)` says the command leaves that folder out (and `root_ok` the root)."""
    import glob
    for target in targets:
        for text in expand(target, state["vars"]):
            path = os.path.expanduser(text)
            if state["cwd"] is not None:
                path = re.sub(r"\$\{?PWD\}?(?![\w])", lambda found: state["cwd"], path)
            if "$" in path or (state["cwd"] is None and not os.path.isabs(path)):
                continue
            full = os.path.normpath(os.path.join(state["cwd"] or "/", path))
            # A glob is judged by every folder it expands to now.
            pattern = path if os.path.isabs(path) else os.path.join(glob.escape(state["cwd"]), path)
            for each in glob.glob(pattern)[:256] if GLOB.search(path) else [full]:
                if not os.path.isdir(each):
                    continue
                for held in PROTECTED:
                    if os.path.isdir(os.path.join(each, held)) and not (leaves_out and leaves_out(text, held)):
                        raise Refuse(what + " reaches " + text + ", which holds a " + held + "/ folder not left out")
                if os.path.exists(os.path.join(each, ".git")) and not root_ok:
                    raise Refuse(what + " would remove from " + text + ", a repository's root")


def positional(args, value_options):
    """The words of a command that are neither options nor the values of the options named."""
    return [arg for i, arg in enumerate(args)
            if not arg.startswith("-") and not (i and args[i - 1] in value_options)]


def remote_command(args):
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if len(args[i]) == 2 and args[i][1] in SSH_VALUE_OPTIONS else 1
    return " ".join(args[i + 1:])


def shell(args, state, fed, feed, depth):
    """Judge the command line a shell is handed: after -c, as a here-string, or down a pipe."""
    for i, arg in enumerate(args):
        if re.match(r"-[A-Za-z]*c[A-Za-z]*$", arg):
            return judge(" ".join(args[i + 1:i + 2]), state, depth + 1, fed, feed)
        if arg.startswith("<<<"):
            return judge(arg[3:] or " ".join(args[i + 1:i + 2]), state, depth + 1, fed, feed)
    if all(arg.startswith("-") for arg in args):
        # Nothing names a script, so the shell reads one from the pipe: a word upstream may be that script.
        for word in feed:
            if re.search(r"\s", word):
                judge(word, dict(state), depth + 1)


def bare(words, variables=None):
    """A simple command without its leading redirects, keywords and assignments; the assignments are recorded."""
    cmd = list(words)
    while cmd:
        redirect = REDIRECT.fullmatch(cmd[0])
        if redirect:
            cmd = cmd[1:] if redirect.group(1) else cmd[2:]
        elif cmd[0] == "function":
            cmd = cmd[2:]
        elif ASSIGNMENT.match(cmd[0]):
            if variables is not None:
                name, value = cmd[0].split("=", 1)
                variables[name.rstrip("+")] = expand(value, variables)
            cmd = cmd[1:]
        elif cmd[0] in KEYWORDS:
            cmd = cmd[1:]
        else:
            break
    return cmd


def recognised(args):
    """Index of the first word a wrapper's command can be: one the guard parses or takes as harmless. Else None."""
    for i, arg in enumerate(args):
        if os.path.basename(arg) in KNOWN + HARMLESS or re.fullmatch(INTERPRETER, os.path.basename(arg)):
            return i
    return None


def only_text(words):
    """True when a command never runs what a pipe hands it."""
    cmd = bare(words)
    if cmd and os.path.basename(cmd[0]) in WRAPPERS:
        i = recognised(cmd[1:])
        return i is not None and only_text(cmd[1 + i:])
    return not cmd or os.path.basename(cmd[0]) in READERS


def unmodelled(name, words, state, fed, feed, depth, quiet):
    """A command the guard does not model may run what it is handed: the crude rule for its own words, then its
    first word the guard knows is judged as its command. Its own options are not modelled."""
    handed = " ".join(words)
    if CRUDE_PATH.search(handed) and (HANDED_VERB.search(handed) or CODE_REMOVE.search(handed)):
        raise Refuse(name + " is not a command the guard reads, and its words hold a removing command and a "
                     "data/, artists/ or outputs/ path")
    i = recognised(words[1:])
    if i is not None:
        command(words[1 + i:], state, fed, feed, depth, quiet)


def command(words, state, fed, feed, depth, quiet=True, bodies=0):
    """Judge one simple command, given as its words; `feed` is the words of the commands piped into it. The removing
    words it cannot run are placed: its own when it only reads them, and with `quiet` (nothing downstream runs
    what it prints) the ones in the text it is handed (`bodies`)."""
    cmd, tally = bare(words, state["vars"]), state["tally"]
    if not cmd:
        return
    name, args, own = os.path.basename(cmd[0]), cmd[1:], occurrences(" ".join(cmd))
    if "$" in name:
        # A command held in a variable: each value the line gives it is judged as the command line it makes. One
        # the guard cannot follow may be anything, text printed on this line included, so nothing said is placed.
        runs = [text for text in expand(cmd[0], state["vars"]) if text != cmd[0]]
        for text in runs:
            aside = dict(state, tally={"placed": 0, "void": False})
            judge(" ".join([text] + [shlex.quote(arg) for arg in args]), aside, depth + 1, fed, feed)
            tally["void"] = tally["void"] or aside["tally"]["void"]
        if not runs or any("$" in os.path.basename((text.split() or ["$"])[0]) for text in runs):
            tally["void"] = True
        unmodelled(name, cmd, state, fed, feed, depth, quiet)
    elif name == "for" and args[1:2] == ["in"]:
        state["vars"][args[0]] = [text for word in args[2:] for text in expand(word, state["vars"])][:64]
        tally["placed"] += own
    elif name in ("cd", "pushd"):
        change_folder(args, state)
        tally["placed"] += own
    elif name in WRAPPERS:
        i = recognised(args)
        if name == "command" and args[:1] in (["-v"], ["-V"]):
            tally["placed"] += own
        elif i is None:
            unmodelled(name, cmd, state, fed, feed, depth, quiet)
        else:
            tally["placed"] += occurrences(" ".join(cmd[:1 + i]))
            command(args[i:], state, fed or name == "xargs", feed, depth, quiet, bodies)
    elif name in SHELLS:
        shell(args, state, fed, feed, depth)
    elif name == "eval":
        judge(" ".join(args), state, depth + 1, fed, feed)
    elif name == "trap":
        judge(" ".join([arg for arg in args if not arg.startswith("-")][:1]), dict(state), depth + 1)
    elif name == "ssh":
        judge(remote_command(args), dict(state, cwd=None, inside=False), depth + 1)
    elif re.fullmatch(INTERPRETER, name):
        # With no script named, the interpreter reads its code from the pipe.
        code = args + (list(feed) if all(arg.startswith("-") for arg in args) else [])
        if any(code_removes(text) for text in code):
            raise Refuse("the code given to " + name + " holds a removing call and a data/, artists/ or outputs/ path")
    elif name == "git":
        git(args, state, fed, feed)
        tally["placed"] += own + bodies
    elif name in REMOVERS:
        removing(name, args, state, fed, feed)
        too_wide(name, [arg for arg in args if not arg.startswith("-")], state)
        finds = [i for i, word in enumerate(feed) if os.path.basename(word) in FINDERS]
        if fed and finds:
            # Names piped in from a find: the find's start is the target, and its prunes do not hold for a pipe.
            upstream = list(feed[finds[-1] + 1:])
            too_wide("find | " + name, find_starts(upstream), state, None, find_narrowed(upstream))
        tally["placed"] += own
    elif name in FINDERS:
        removes = "-delete" in args
        if removes:
            removing("find -delete", find_targets(args), state, fed, feed, globs=False)
        for i, arg in enumerate(args):
            if arg in ("-exec", "-execdir", "-ok", "-okdir"):
                run = args[i + 1:]
                end = min([run.index(stop) for stop in (";", "+") if stop in run] or [len(run)])
                removes = removes or runs_remover(run[:end])
                # The command find runs is fed its paths by find, and places its own words.
                own -= occurrences(" ".join(run[:end]))
                command(run[:end], state, True, find_targets(args[:i]), depth)
        if removes:
            too_wide("find", find_starts(args), state, lambda start, held: find_leaves_out(args, start, held),
                     find_narrowed(args))
        tally["placed"] += own
    elif name == "rsync":
        if any(arg.startswith("--del") for arg in args):
            removing("rsync --delete", positional(args, RSYNC_VALUE_OPTIONS), state, fed, feed)
            ends = positional(args, RSYNC_VALUE_OPTIONS)[1:][-1:]
            too_wide("rsync --delete", ends, state, lambda end, held: rsync_leaves_out(args, held))
        tally["placed"] += own
    elif name not in HARMLESS:
        unmodelled(name, cmd, state, fed, feed, depth, quiet)
    elif quiet:
        tally["placed"] += own + bodies


def judge(cmd, state, depth, fed=False, feed=()):
    """Raise Refuse when a command line removes artist material; ValueError when it cannot be read as shell."""
    if depth > 8:
        raise Refuse("the command is nested too deep to read")
    pieces, nested, _ = scan(cmd)
    lines = [(shlex.split(text), piped, comments, bodies) for text, piped, comments, bodies in pieces]
    for i, (words, piped, _, _) in enumerate(lines):
        # What a loop or a group printed is piped into something that may run it: nothing said in it is placed.
        if piped and not only_text(words) and (not lines[i - 1][0] or lines[i - 1][0][0] in CLOSERS):
            state["tally"]["void"] = True
    upstream = list(feed)
    for i, (words, piped, comments, bodies) in enumerate(lines):
        if not piped:
            upstream = list(feed)
        state["tally"]["placed"] += comments
        rest = i + 1
        while rest < len(lines) and lines[rest][1] and only_text(lines[rest][0]):
            rest += 1
        quiet = rest == len(lines) or not lines[rest][1]
        command(words, state, fed, upstream, depth, quiet, bodies)
        upstream += words
    for inner in nested:
        judge(inner, dict(state), depth + 1)


def judge_bash(cmd, cwd, bases):
    try:
        inside = any(protected_under(os.path.join(full, "x"), bases) for full in {cwd, os.path.realpath(cwd)})
        tally = {"placed": 0, "void": False}
        judge(cmd, {"cwd": cwd, "inside": bool(inside), "vars": {}, "text": cmd, "tally": tally}, 0)
        # The floor: a removing word the guard could not place, on a line that holds a protected path.
        if CODE_PATH.search(cmd) and occurrences(cmd) > (0 if tally["void"] else tally["placed"]):
            raise Refuse("the line holds a removing word the guard cannot place, and a data/, artists/ or outputs/ "
                         "path")
    except RecursionError:
        raise Refuse("the command is nested too deep to read")
    except Refuse as refusal:
        raise Refuse(str(refusal) + "; artist material is never removed from a session")
    except Exception:
        # Not readable as shell: the crude rule, a removing word and a protected path anywhere in the text.
        if CRUDE_VERB.search(cmd) and CRUDE_PATH.search(cmd):
            raise Refuse("the command could not be read, and it holds a removing command and a data/, artists/ or "
                         "outputs/ path; artist material is never removed from a session")


def verdict(call, root, repos=(), siblings_read_only=False):
    """Why a tool call (the hook's JSON, parsed) is refused, or None. No `repos`: the root is the repository."""
    root = os.path.abspath(root)
    homes = [os.path.join(root, repo) for repo in repos] or [root]
    bases = sorted(set(homes) | {os.path.realpath(home) for home in homes})
    try:
        tool, args = call.get("tool_name", ""), call.get("tool_input") or {}
        cwd = call.get("cwd") or root
        if tool == "Bash":
            target = args.get("command", "")
        else:
            target = args.get("file_path") or args.get("notebook_path") or ""
        if not isinstance(target, str) or not isinstance(cwd, str):
            raise ValueError("not text")
    except Exception:
        return "the tool call could not be read"
    try:
        if tool == "Bash":
            judge_bash(target, cwd, bases)
        elif target:
            judge_edit(target, cwd, bases)
            if siblings_read_only:
                judge_sibling(target, cwd, root)
    except Refuse as refusal:
        return str(refusal)
    except Exception:
        return "the path could not be read"
    return None


def main():
    try:
        call = json.load(sys.stdin)
    except Exception:
        call = None
    repos = [arg for arg in sys.argv[2:] if not arg.startswith("--")]
    why = verdict(call, sys.argv[1], repos, "--siblings-read-only" in sys.argv[2:])
    if why:
        print("refused: " + why, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
