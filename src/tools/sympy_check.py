# File: src/tools/sympy_check.py
# Created: 2026-11-18
# Created by: Lathika
# Purpose: Run a short numeric test written by a model (SymPy / NumPy code that checks a
#          claimed identity at random points) in a restricted subprocess with a timeout.
#          The code must print PASS or FAIL as its last line. Result: pass, fail or error.
#
#          Safety layers (simple, not a security boundary):
#          1. the code is parsed first and refused if it imports anything but math,
#             numpy and sympy style modules, calls open/exec/eval, or touches dunder
#             attributes or numpy file functions;
#          2. it runs in a separate Python process in an empty temporary folder, with a
#             clean environment, a CPU limit, a memory limit, a file-size limit, and a
#             wall-clock timeout.

import ast
import os
import subprocess
import sys
import tempfile

TIMEOUT = 10                     # seconds of wall-clock time for one test
MEMORY_LIMIT = 3 * 1024 ** 3     # bytes of address space (numpy needs some headroom)
MAX_OUTPUT = 300                 # characters of detail kept

ALLOWED_MODULES = {"math", "cmath", "random", "fractions", "itertools", "numpy", "sympy"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals",
                "vars", "getattr", "setattr", "delattr", "breakpoint", "exit", "quit"}
BANNED_ATTRIBUTES = {"load", "loadtxt", "genfromtxt", "fromfile", "tofile", "save", "savetxt",
                     "savez", "savez_compressed", "memmap", "system", "popen", "lambdify"}

# Runs inside the child process: set the limits, then run the test file.
LAUNCHER = (
    "import resource, runpy, sys\n"
    "cpu, mem = int(sys.argv[2]), int(sys.argv[3])\n"
    "resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))\n"
    "resource.setrlimit(resource.RLIMIT_AS, (mem, mem))\n"
    "resource.setrlimit(resource.RLIMIT_FSIZE, (1000000, 1000000))\n"
    "runpy.run_path(sys.argv[1], run_name='__main__')\n"
)


def refuse_reason(code):
    """Why this code may not run, or "" if it looks fine."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        else:
            roots = []
        for root in roots:
            if root not in ALLOWED_MODULES:
                return f"import of '{root}' is not allowed"
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"use of '{node.id}' is not allowed"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in BANNED_ATTRIBUTES:
                return f"use of '.{node.attr}' is not allowed"
    return ""


def read_output(stdout):
    """Turn the last printed line into (status, detail)."""
    lines = [line.strip() for line in stdout.strip().splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    if last.upper().startswith("PASS"):
        return "pass", last[:MAX_OUTPUT]
    if last.upper().startswith("FAIL"):
        return "fail", last[:MAX_OUTPUT]
    return "error", "the test did not print PASS or FAIL"


def run_test(code, timeout=TIMEOUT):
    """Run model-written test code. Returns {"status": "pass" | "fail" | "error", "detail": str}."""
    reason = refuse_reason(code)
    if reason:
        return {"status": "error", "detail": reason}

    env = {"PATH": os.environ.get("PATH", ""),
           "PYTHONPATH": os.environ.get("PYTHONPATH", ""),     # where the packages live
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory() as folder:
        script = os.path.join(folder, "claim_test.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(code)
        command = [sys.executable, "-B", "-c", LAUNCHER, script,
                   str(timeout + 2), str(MEMORY_LIMIT)]
        try:
            done = subprocess.run(command, cwd=folder, env=env, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "error", "detail": f"timed out after {timeout} s"}

    status, detail = read_output(done.stdout)
    if status == "error" and done.returncode != 0:
        last_line = done.stderr.strip().splitlines()[-1:] or ["the test crashed"]
        detail = last_line[0][:MAX_OUTPUT]
    return {"status": status, "detail": detail}