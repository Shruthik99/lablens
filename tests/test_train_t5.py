import json
import subprocess

from lablens.train_t5 import (
    build_run_info,
    get_git_commit,
    parse_args,
    sha256_file,
    write_run_info,
)


def test_parse_args_defaults():
    args = parse_args([])

    assert args.data == "data/t5"
    assert args.model == "google/flan-t5-base"
    assert args.out == "models/flan-t5"
    assert args.max_input == 768
    assert args.max_target == 256
    assert args.epochs == 3
    assert args.batch == 8
    assert args.grad_accum == 2
    assert args.lr == 3e-4
    assert args.warmup == 0.06
    assert args.seed == 42
    assert args.limit is None
    assert args.max_steps == -1
    assert args.fp16 is False
    assert args.gradient_checkpointing is True
    assert args.git_commit is None


def test_parse_args_overrides():
    args = parse_args(
        [
            "--data",
            "custom/data",
            "--model",
            "google/flan-t5-small",
            "--out",
            "models/smoke",
            "--max-input",
            "128",
            "--max-target",
            "32",
            "--epochs",
            "1",
            "--batch",
            "2",
            "--grad-accum",
            "1",
            "--lr",
            "1e-4",
            "--warmup",
            "0.1",
            "--seed",
            "7",
            "--limit",
            "16",
            "--max-steps",
            "4",
            "--fp16",
            "--no-gradient-checkpointing",
            "--git-commit",
            "deadbeef",
        ]
    )

    assert args.data == "custom/data"
    assert args.model == "google/flan-t5-small"
    assert args.out == "models/smoke"
    assert args.max_input == 128
    assert args.max_target == 32
    assert args.epochs == 1
    assert args.batch == 2
    assert args.grad_accum == 1
    assert args.lr == 1e-4
    assert args.warmup == 0.1
    assert args.seed == 7
    assert args.limit == 16
    assert args.max_steps == 4
    assert args.fp16 is True
    assert args.gradient_checkpointing is False
    assert args.git_commit == "deadbeef"


def test_sha256_file_matches_known_content(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("hello world", encoding="utf-8")

    assert sha256_file(path) == hashlib_sha256_hex("hello world")


def test_sha256_file_differs_for_different_content(tmp_path):
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    path_a.write_text("one", encoding="utf-8")
    path_b.write_text("two", encoding="utf-8")

    assert sha256_file(path_a) != sha256_file(path_b)


def test_get_git_commit_returns_none_when_git_missing(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(subprocess, "run", _raise)

    assert get_git_commit() is None


def test_get_git_commit_returns_none_when_not_a_repo(monkeypatch):
    def _raise(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(subprocess, "run", _raise)

    assert get_git_commit() is None


def test_get_git_commit_returns_stripped_hash(monkeypatch):
    class _Result:
        stdout = "abc123\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Result())

    assert get_git_commit() == "abc123"


def test_build_run_info_contains_all_settings_and_results(tmp_path):
    args = parse_args(["--limit", "16", "--max-steps", "4", "--model", "google/flan-t5-small"])
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text('{"input": "a", "target": "b"}\n', encoding="utf-8")
    val_path.write_text('{"input": "c", "target": "d"}\n', encoding="utf-8")

    info = build_run_info(
        args,
        train_path,
        val_path,
        device="cpu",
        fp16=False,
        git_commit="abc123",
        train_loss=1.23,
        val_loss=4.56,
        training_seconds=12.5,
    )

    assert info["settings"]["model"] == "google/flan-t5-small"
    assert info["settings"]["limit"] == 16
    assert info["settings"]["max_steps"] == 4
    assert info["settings"]["fp16"] is False
    assert info["settings"]["gradient_checkpointing"] is True
    assert info["settings"]["git_commit_override"] is None
    assert info["data_sha256"]["train.jsonl"] == sha256_file(train_path)
    assert info["data_sha256"]["val.jsonl"] == sha256_file(val_path)
    assert info["git_commit"] == "abc123"
    assert info["device"] == "cpu"
    assert info["train_loss"] == 1.23
    assert info["val_loss"] == 4.56
    assert info["training_seconds"] == 12.5


def test_build_run_info_records_git_commit_override(tmp_path):
    args = parse_args(["--git-commit", "override-sha"])
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    train_path.write_text('{"input": "a", "target": "b"}\n', encoding="utf-8")
    val_path.write_text('{"input": "c", "target": "d"}\n', encoding="utf-8")

    info = build_run_info(
        args,
        train_path,
        val_path,
        device="cpu",
        fp16=False,
        git_commit="override-sha",
        train_loss=1.0,
        val_loss=2.0,
        training_seconds=1.0,
    )

    assert info["settings"]["git_commit_override"] == "override-sha"
    assert info["git_commit"] == "override-sha"


def test_write_run_info_writes_valid_json(tmp_path):
    path = tmp_path / "out" / "run_info.json"
    info = {"a": 1, "nested": {"b": 2}}

    write_run_info(path, info)

    assert json.loads(path.read_text(encoding="utf-8")) == info


def hashlib_sha256_hex(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
