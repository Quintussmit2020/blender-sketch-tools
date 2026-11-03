"""Headless test entry point.

blender.exe --background --factory-startup --python sketch_tools/tests/run.py

Discovers every sketch_tools/tests/test_*.py, runs each test_* function, prints one line per
test and a summary, and exits non-zero on any failure."""

import importlib
import inspect
import os
import sys
import traceback

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    passed = 0
    failed = 0
    for name in sorted(os.listdir(TESTS_DIR)):
        if not (name.startswith("test_") and name.endswith(".py")):
            continue
        mod_name = "sketch_tools.tests." + name[:-3]
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            failed += 1
            print(f"FAIL {mod_name} (import)")
            traceback.print_exc()
            continue
        tests = [
            (fn_name, fn) for fn_name, fn in inspect.getmembers(module, inspect.isfunction)
            if fn_name.startswith("test_") and fn.__module__ == module.__name__
        ]
        tests.sort(key=lambda item: item[1].__code__.co_firstlineno)
        for fn_name, fn in tests:
            label = f"{name[:-3]}.{fn_name}"
            try:
                fn()
            except Exception:
                failed += 1
                print(f"FAIL {label}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"PASS {label}")

    print(f"SUMMARY: {passed} passed, {failed} failed, {passed + failed} total")
    sys.stdout.flush()
    if failed:
        sys.exit(1)


main()
