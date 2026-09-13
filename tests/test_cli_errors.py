import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_src(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the project CLI and capture its output."""
    return subprocess.run(
        [sys.executable, "-m", "src", *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


with tempfile.TemporaryDirectory() as temp_directory:
    temp_path = Path(temp_directory)
    missing_file = temp_path / "missing.json"
    result = run_src([
        "--functions_definition",
        str(missing_file),
    ])
    assert result.returncode != 0
    assert "Error:" in result.stdout
    assert "Traceback" not in result.stderr
    malformed = temp_path / "malformed.json"
    malformed.write_text("[{invalid json]", encoding="utf-8")
    result = run_src([
        "--functions_definition",
        str(malformed),
    ])
    assert result.returncode != 0
    assert "Error:" in result.stdout
    assert "Traceback" not in result.stderr
    wrong_root = temp_path / "wrong_root.json"
    wrong_root.write_text(
        json.dumps({"name": "not a list"}),
        encoding="utf-8",
    )
    result = run_src([
        "--functions_definition",
        str(wrong_root),
    ])
    assert result.returncode != 0
    assert "The JSON root must be a list." in result.stdout
    assert "Traceback" not in result.stderr
    empty_functions = temp_path / "empty_functions.json"
    empty_functions.write_text("[]", encoding="utf-8")
    result = run_src([
        "--functions_definition",
        str(empty_functions),
    ])
    assert result.returncode != 0
    assert "No function definitions were provided." in result.stdout
    assert "Traceback" not in result.stderr
    result = run_src([
        "--limit",
        "0",
    ])
    assert result.returncode != 0
    assert "--limit must be greater than zero." in result.stdout
    assert "Traceback" not in result.stderr
print("CLI error handling tests passed.")
