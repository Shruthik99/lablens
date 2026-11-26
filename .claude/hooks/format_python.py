import json
import os
import subprocess
import sys
from pathlib import Path


def project_root() -> str:
    return os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path")
    if not file_path or not file_path.endswith(".py"):
        return

    root = project_root()
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(root) / path

    if not path.exists():
        return

    for cmd in (
        [sys.executable, "-m", "ruff", "check", "--fix", str(path)],
        [sys.executable, "-m", "black", str(path)],
    ):
        try:
            subprocess.run(cmd, cwd=root, capture_output=True, timeout=60, check=False)
        except (OSError, subprocess.SubprocessError):
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - this hook must never fail the session
        print(f"format_python.py: unexpected error, skipping formatting: {exc}", file=sys.stderr)
    sys.exit(0)
