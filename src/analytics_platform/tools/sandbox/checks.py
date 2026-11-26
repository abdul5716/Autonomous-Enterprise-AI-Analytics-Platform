"""Static checks on generated analysis code, before it reaches the sandbox.

The container is the real security boundary (no network, read-only, no capabilities); these
checks reject obviously unsafe or out-of-scope code early, with messages the LLM can act on.
"""

import ast

ALLOWED_IMPORTS = {
    "pandas", "numpy", "scipy", "math", "statistics", "json", "datetime", "collections",
    "itertools", "functools", "re", "decimal", "fractions", "operator", "string",
}  # fmt: skip

BLOCKED_CALLS = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input",
    "breakpoint", "getattr", "setattr", "delattr", "memoryview", "help", "exit", "quit",
}  # fmt: skip

# pandas/numpy I/O: everything the analysis needs is already in memory.
BLOCKED_ATTRIBUTE_PREFIXES = ("read_",)
BLOCKED_ATTRIBUTES = {
    "to_csv", "to_excel", "to_parquet", "to_pickle", "to_sql", "to_hdf", "to_feather",
    "to_clipboard", "to_stata", "to_orc", "load", "loadtxt", "genfromtxt", "fromfile",
    "tofile", "save", "savez", "savetxt", "memmap", "system", "popen",
}  # fmt: skip

MAX_CODE_CHARS = 8_000


def check_code(code: str) -> list[str]:
    if len(code) > MAX_CODE_CHARS:
        return [f"Code is too long ({len(code)} > {MAX_CODE_CHARS} characters)."]
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error on line {exc.lineno}: {exc.msg}"]
    problems: list[str] = []
    assigns_result = False
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            for name in names:
                root = (name or "").split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    problems.append(f"Line {line}: import of '{name}' is not allowed.")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS:
                problems.append(f"Line {line}: calling '{node.func.id}' is not allowed.")
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__"):
                problems.append(f"Line {line}: attribute '{attr}' is not allowed.")
            elif attr in BLOCKED_ATTRIBUTES or attr.startswith(BLOCKED_ATTRIBUTE_PREFIXES):
                problems.append(f"Line {line}: file I/O ('{attr}') is not allowed.")
        elif isinstance(node, ast.Name) and node.id.startswith("__") and node.id != "__name__":
            problems.append(f"Line {line}: name '{node.id}' is not allowed.")
        elif isinstance(node, ast.Global | ast.Nonlocal):
            problems.append(f"Line {line}: global/nonlocal statements are not allowed.")
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigns_result |= any(isinstance(t, ast.Name) and t.id == "result" for t in targets)
    if not assigns_result:
        problems.append("The code must assign its answer to a variable named `result`.")
    return problems
