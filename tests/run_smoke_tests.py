#!/usr/bin/env python3
import argparse
import importlib.util
import inspect
import tempfile
from pathlib import Path
from unittest import SkipTest


def _load_module(path):
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _test_files(test_dir):
    return sorted(
        path
        for path in test_dir.glob("test_*.py")
        if path.name != Path(__file__).name
    )


def _call_test(func):
    kwargs = {}
    temp_dirs = []
    try:
        for name in inspect.signature(func).parameters:
            if name == "tmp_path":
                temp_dir = tempfile.TemporaryDirectory()
                temp_dirs.append(temp_dir)
                kwargs[name] = Path(temp_dir.name)
            else:
                raise SkipTest(f"Unsupported fixture argument: {name}")
        func(**kwargs)
    finally:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Run lightweight PyQUDA Measurement smoke tests.")
    parser.add_argument("--include-optional", action="store_true", help="Also run tests marked as GPU or external-HDF5 dependent.")
    args = parser.parse_args()

    test_dir = Path(__file__).resolve().parent
    passed = 0
    skipped = 0
    failed = 0

    for path in _test_files(test_dir):
        try:
            module = _load_module(path)
        except SkipTest as err:
            skipped += 1
            print(f"SKIP {path.name}: {err}")
            continue

        requirement = getattr(module, "TEST_REQUIRES", None)
        if requirement and not args.include_optional:
            skipped += 1
            print(f"SKIP {path.name}: optional requirement {requirement!r}")
            continue

        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            func = getattr(module, name)
            if not callable(func):
                continue
            label = f"{path.name}::{name}"
            try:
                _call_test(func)
            except SkipTest as err:
                skipped += 1
                print(f"SKIP {label}: {err}")
            except Exception as err:
                failed += 1
                print(f"FAIL {label}: {err}")
            else:
                passed += 1
                print(f"PASS {label}")

    print(f"SUMMARY passed={passed} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
