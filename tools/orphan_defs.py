"""Zero-LLM, read-only census of module-level orphans: defs, constants, config keys.

The companion to `tools/orphan_gates.py`. That one asks "does this gate still
fire?"; this one asks the cheaper question underneath it — "does anything still
NAME this?" Both exist because v1's deletion left ~2.4k lines and 133 config keys
that no longer had a caller, and neither `grep` nor a test run says so.

    python tools/orphan_defs.py                # module-level defs
    python tools/orphan_defs.py --constants    # + module-level assignments
    python tools/orphan_defs.py --config       # config keys with no reader

**Token grep is the wrong instrument for the def scan.** v2's docstrings quote v1
function names by the dozen, and prose in a docstring counts as a reference, which
hides exactly the functions the deletion orphaned. References come from the AST;
docstrings never contribute.

Five predicates, each of which gave a WRONG answer before it was encoded:

1. **Intra-module references count.** The first run reported 490 defs / 15k lines
   because it excluded the defining file, so every private helper called only by
   its own module (`_shard_dir`, 5 self-refs) read as an orphan.
2. **`ast.Store` does not count.** A constant's own `X = ...` target IS an
   `ast.Name`, so counting it made every unused constant look self-referenced and
   the first `--constants` run reported a clean zero.
3. **`unittest` discovers `TestCase` subclasses BY NAME**; nothing references
   them, so a test class is never an orphan.
4. **"Referenced only by its own tests" is dead code with an alibi** — the test
   keeps it green forever and nothing ships it. Reported separately as TESTS-ONLY
   rather than counted as live: that is how the whole chapter-title feature
   (refine + apply + config key) stayed invisible through two audits.
5. **A config key read through an f-string has no literal to grep.**
   `config.py:512` reads `api.get(f"{role}_base_url")`, so all 29
   `{role}_{base_url,api_key,keys,model,thinking_mode,…}` keys look dead and are
   not. `--config` whitelists that family explicitly; ANY new interpolated key
   family must be added to `INTERPOLATED` or this tool will recommend deleting it.

The output is a bug report, not a deletion list — the same rule CLAUDE.md states
for silent gates. A dead def can be v1 machinery v2 replaced (delete) or a
capability whose feeder is unwired (fix the wiring). `--config` cannot tell those
apart; only reading the consumer can.
"""
from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"__pycache__", ".git", "novels", "experiments", "telemetry", "benchmarks"}

# Config-key families read via f-string interpolation (predicate 5).
_ROLES = ("planning", "writing", "extraction", "review", "refine")
_SUFFIXES = ("base_url", "api_key", "keys", "model", "thinking_mode",
             "thinking_budget_tokens", "reasoning_effort")
INTERPOLATED = {f"{r}_{s}" for r in _ROLES for s in _SUFFIXES}


def py_files() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.py")
                  if not any(part in SKIP for part in p.parts))


def module_of(p: Path) -> str:
    return ".".join(p.relative_to(ROOT).with_suffix("").parts)


def _parse(files: list[Path]) -> dict[Path, ast.Module]:
    trees = {}
    for p in files:
        try:
            trees[p] = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            print(f"!! parse failed {p}: {exc}")
    return trees


def _required_span() -> range:
    """Line range of `load_config`'s `required` dict — a declaration, not a read.

    Every key is spelled there, so counting it would make the whole config
    inventory look live.
    """
    src = (ROOT / "config.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if getattr(n, "name", "") == "load_config")
    req = next(n for n in ast.walk(fn)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "required"
                       for t in n.targets))
    return range(req.lineno, req.end_lineno + 1)


def scan_defs(constants: bool) -> int:
    files = py_files()
    trees = _parse(files)

    defs: dict[str, list[tuple[Path, int, int]]] = defaultdict(list)
    for p, tree in trees.items():
        for node in tree.body:          # module level only: a method is reachable
            size = getattr(node, "end_lineno", node.lineno) - node.lineno + 1
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                defs[node.name].append((p, node.lineno, size))
            elif isinstance(node, ast.Assign) and constants:
                # A deleted function's prompt string outlives it silently, and a
                # 48-line SYSTEM prompt nobody sends is the most misleading kind
                # of leftover: it reads as live doctrine.
                for t in node.targets:
                    if isinstance(t, ast.Name) and not t.id.startswith("__"):
                        defs[t.id].append((p, node.lineno, size))

    refs: dict[str, set[Path]] = defaultdict(set)
    for p, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                refs[node.id].add(p)
            elif isinstance(node, ast.Attribute):
                refs[node.attr].add(p)
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    refs[a.name].add(p)

    rows, tests_only = [], []
    for name, sites in sorted(defs.items()):
        if len(sites) > 1:               # same name in two modules: hand-check
            continue
        p, line, size = sites[0]
        users = refs[name] - {p}
        prod = {q for q in users if "tests" not in q.parts}
        if prod:
            continue
        # predicates 1 + 2: anything naming it inside its own module, LOADS only
        own = sum(1 for n in ast.walk(trees[p])
                  if (isinstance(n, ast.Name) and n.id == name
                      and isinstance(n.ctx, ast.Load))
                  or (isinstance(n, ast.Attribute) and n.attr == name))
        if own:
            continue
        if users:                       # predicate 4
            tests_only.append((size, module_of(p), name, line))
            continue
        if "tests" in p.parts and any(   # predicate 3
                isinstance(n, ast.ClassDef) and n.name == name
                for n in trees[p].body):
            continue
        rows.append((size, module_of(p), name, line))

    for size, mod, name, line in sorted(tests_only, reverse=True):
        print(f"TESTS-ONLY  {mod:<20} {name}:{line}  ({size} lines)")
    rows.sort(reverse=True)
    print(f"\n{'lines':>6}  {'module':<20} name")
    by_mod: dict[str, int] = defaultdict(int)
    for size, mod, name, line in rows:
        by_mod[mod] += size
        print(f"{size:>6}  {mod:<20} {name}:{line}")
    print(f"\n{len(rows)} orphan defs, {sum(r[0] for r in rows)} lines")
    for mod, n in sorted(by_mod.items(), key=lambda kv: -kv[1]):
        print(f"  {mod:<20} {n}")
    return 0


def scan_config() -> int:
    files = py_files()
    texts = {p: p.read_text(encoding="utf-8", errors="replace") for p in files}
    req_span = _required_span()

    tpl = ROOT / "config_template.example.yaml"      # tracked, credential-free
    keys: list[str] = []
    for line in tpl.read_text(encoding="utf-8").split("\n"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*:", s)
        if m:
            keys.append(m.group(1))

    dead = []
    for k in dict.fromkeys(keys):
        if k in INTERPOLATED:
            continue
        users = set()
        for p, t in texts.items():
            hits = list(re.finditer(r'["\']' + re.escape(k) + r'["\']', t))
            # attribute form covers `paths.chapters_dir`, which is a dataclass
            # field rather than a dict key
            hits += list(re.finditer(r'\.' + re.escape(k) + r'\b', t))
            for m in hits:
                if p.name == "config.py" and \
                        t.count("\n", 0, m.start()) + 1 in req_span:
                    continue
                users.add(p.name)
        # compare.py's RELEASE_RULE_KEYS names keys to FLAG in an A/B, which is a
        # guard list rather than a reader — a key only it mentions is still dead.
        if not users or users <= {"compare.py"}:
            dead.append((k, sorted(users)))
    for k, u in dead:
        print(f"  {k:<44} {u}")
    print(f"\n{len(dead)} of {len(dict.fromkeys(keys))} template keys have no reader")
    return 0


def main() -> int:
    if "--config" in sys.argv:
        return scan_config()
    return scan_defs("--constants" in sys.argv)


if __name__ == "__main__":
    sys.exit(main())
