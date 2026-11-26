import json
import os
import sys
from pathlib import Path

READ_AND_WRITE_TOOLS = {"Read", "Edit", "Write", "MultiEdit"}
WRITE_TOOLS = {"Edit", "Write", "MultiEdit"}


def project_root() -> Path:
    root = os.environ.get("CLAUDE_PROJECT_DIR")
    return Path(root).resolve() if root else Path.cwd().resolve()


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path")
    if not file_path:
        sys.exit(0)

    root = project_root()
    path = Path(file_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()

    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = path.as_posix()

    is_env_file = path.name == ".env"
    in_data = rel == "data" or rel.startswith("data/")
    in_fixtures = rel == "tests/fixtures" or rel.startswith("tests/fixtures/")

    if (is_env_file or in_data) and tool_name in READ_AND_WRITE_TOOLS:
        print(
            f"Blocked by protect_files.py: '{rel}' is protected (inside data/ or a .env file) "
            f"and cannot be read, edited, or written.",
            file=sys.stderr,
        )
        sys.exit(2)

    if in_fixtures and tool_name in WRITE_TOOLS:
        print(
            f"Blocked by protect_files.py: '{rel}' is inside tests/fixtures/ "
            f"and cannot be edited or written (reading is allowed).",
            file=sys.stderr,
        )
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
