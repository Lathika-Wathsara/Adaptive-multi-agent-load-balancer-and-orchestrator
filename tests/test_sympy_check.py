# File: tests/test_sympy_check.py
# Created: 2026-11-18
# Created by: Lathika
# Purpose: Tests for src/tools/sympy_check.py: the restricted subprocess that runs a
#          model-written numeric test (pass, fail, crash, timeout, refused code).

import time

from src.tools import sympy_check as sc

PASSING = """
import numpy as np
rng = np.random.default_rng(0)
a = rng.normal(size=5) + 1j * rng.normal(size=5)
b = rng.normal(size=5) + 1j * rng.normal(size=5)
left = abs(a - b) ** 2
right = abs(a) ** 2 + abs(b) ** 2 - 2 * (a * b.conj()).real
print("PASS" if np.allclose(left, right) else f"FAIL {left} {right}")
"""

FAILING = PASSING.replace("- 2 * (a", "+ 2 * (a")


def test_correct_identity_passes():
    result = sc.run_test(PASSING)
    assert result["status"] == "pass", result


def test_wrong_identity_fails():
    result = sc.run_test(FAILING)
    assert result["status"] == "fail"
    assert result["detail"].startswith("FAIL")


def test_sympy_test_runs():
    code = ("import sympy as sp\nx = sp.symbols('x')\n"
            "print('PASS' if sp.simplify(sp.sin(x)**2 + sp.cos(x)**2 - 1) == 0 else 'FAIL')")
    assert sc.run_test(code)["status"] == "pass"


def test_no_pass_or_fail_line_is_an_error():
    result = sc.run_test("print('hello')")
    assert result["status"] == "error"
    assert "PASS or FAIL" in result["detail"]


def test_crash_is_an_error_with_the_reason():
    result = sc.run_test("import numpy as np\nnp.linalg.inv(np.zeros((2, 2)))")
    assert result["status"] == "error"
    assert "Singular" in result["detail"]


def test_syntax_error_is_refused():
    result = sc.run_test("print('PASS'")
    assert result["status"] == "error"
    assert "syntax" in result["detail"]


def test_endless_loop_times_out():
    start = time.time()
    result = sc.run_test("while True:\n    pass", timeout=2)
    assert result["status"] == "error"
    assert time.time() - start < 8


def test_refused_code_never_runs():
    bad = ["import os\nprint('PASS')",
           "import subprocess\nprint('PASS')",
           "from socket import socket\nprint('PASS')",
           "print(open('/etc/passwd').read())",
           "exec('print(1)')",
           "eval('1+1')",
           "__import__('os')",
           "x = ().__class__.__bases__\nprint('PASS')",
           "import numpy as np\nnp.loadtxt('/etc/passwd')",
           "import numpy as np\nnp.save('x.npy', np.zeros(3))"]
    for code in bad:
        result = sc.run_test(code)
        assert result["status"] == "error", code
        assert "not allowed" in result["detail"], code


def test_memory_hog_is_stopped():
    result = sc.run_test("import numpy as np\nx = np.ones((20000, 20000))\nprint('PASS')")
    assert result["status"] == "error"


def test_values_printed_after_the_verdict_line():
    result = sc.run_test("print('FAIL')\nprint(1.5, 2.5)")
    assert result["status"] == "fail" and "1.5" in result["detail"]
    result = sc.run_test("print('PASS')\nprint('checked 5 points')")
    assert result["status"] == "pass"