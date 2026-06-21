#!/usr/bin/env python3
"""Project-Lewis environment checker for new contributors.

Uses only the Python standard library (shutil, subprocess, sys).
"""

import shutil
import subprocess
import sys

HEADER = "Project-Lewis — Environment Check"

REQUIRED_TOOLS = [
    ("uv", ["uv", "--version"]),
    ("docker", ["docker", "--version"]),
    ("git", ["git", "--version"]),
    ("make", ["make", "--version"]),
]

OPTIONAL_TOOLS = [
    ("arm-none-eabi-gcc", ["arm-none-eabi-gcc", "--version"]),
]


def _first_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else ""


def check_python() -> bool:
    """Check that the running interpreter is Python 3.12.x."""
    version_info = sys.version_info
    version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    if version_info.major == 3 and version_info.minor == 12:
        print(f"  [OK] Python {version} (3.12.x)")
        return True
    print(f"  [FAIL] Python {version} (expected 3.12.x)")
    return False


def check_required(name: str, version_cmd: list[str]) -> bool:
    """Check a required tool and print its version when available."""
    path = shutil.which(name)
    if path is None:
        print(f"  [FAIL] {name}: not found in PATH")
        return False

    try:
        result = subprocess.run(
            version_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout if result.stdout else result.stderr
        version = _first_line(output)
        if version:
            print(f"  [OK] {name}: {version}")
        else:
            print(f"  [OK] {name}: found at {path}")
        return True
    except Exception as exc:  # noqa: BLE001 - keep going, report failure
        print(f"  [FAIL] {name}: version check failed ({exc})")
        return False


def check_optional(name: str, version_cmd: list[str]) -> bool:
    """Check an optional tool; does not make the overall check fail."""
    path = shutil.which(name)
    if path is None:
        print(f"  [INFO] {name}: not found (optional)")
        return True

    try:
        result = subprocess.run(
            version_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout if result.stdout else result.stderr
        version = _first_line(output)
        if version:
            print(f"  [OK] {name}: {version}")
        else:
            print(f"  [OK] {name}: found at {path}")
    except Exception as exc:  # noqa: BLE001 - optional, keep going
        print(f"  [INFO] {name}: found at {path} (version check failed: {exc})")
    return True


def main() -> int:
    print(HEADER)
    print("-" * len(HEADER))

    ok = True
    ok &= check_python()

    for name, cmd in REQUIRED_TOOLS:
        ok &= check_required(name, cmd)

    for name, cmd in OPTIONAL_TOOLS:
        check_optional(name, cmd)

    print("-" * len(HEADER))
    if ok:
        print("Environment OK")
        return 0

    print("Environment check FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
