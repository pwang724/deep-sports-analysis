"""Ask Astra (gpt-6-astra) about images through the local Codex CLI.

Each call runs `codex exec` headless in a read-only sandbox, in an empty
temporary directory, with a JSON output schema. Answers are cached on disk by a
hash of model, reasoning effort, prompt, schema and image bytes, so a rerun
only pays for items whose inputs changed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

MODEL = "gpt-6-astra"
EFFORT = "medium"  # reasoning effort; overrides the high default in ~/.codex/config.toml


def cache_key(model: str, effort: str, prompt: str, schema: dict, images: list[Path]) -> str:
    h = hashlib.sha256()
    for part in (model, effort, prompt, json.dumps(schema, sort_keys=True)):
        h.update(part.encode())
        h.update(b"\0")
    for p in images:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:24]


def ask(prompt: str, images: list[Path], schema: dict, cache_dir: str | Path,
        model: str = MODEL, effort: str = EFFORT, timeout: float = 600, retries: int = 2) -> tuple[dict, float]:
    """Return (answer matching `schema`, seconds spent; 0 when cached)."""
    cache = Path(cache_dir) / f"{cache_key(model, effort, prompt, schema, images)}.json"
    if cache.exists():
        return json.loads(cache.read_text()), 0.0
    cache.parent.mkdir(parents=True, exist_ok=True)
    t0, err = time.time(), None
    with tempfile.TemporaryDirectory() as tmp:
        schema_path, out = Path(tmp) / "schema.json", Path(tmp) / "out.json"
        schema_path.write_text(json.dumps(schema))
        cmd = ["codex", "exec", "-m", model, "-c", f"model_reasoning_effort={effort}",
               "-s", "read-only", "--skip-git-repo-check", "-C", tmp]
        for p in images:
            cmd += ["-i", str(Path(p).resolve())]
        cmd += ["--output-schema", str(schema_path), "-o", str(out), "-"]  # prompt on stdin
        for _ in range(retries + 1):
            try:
                r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                err = f"timed out after {timeout}s"
                continue
            if r.returncode == 0 and out.exists():
                try:
                    answer = json.loads(out.read_text())
                    break
                except json.JSONDecodeError as e:
                    err = e
            else:
                err = r.stderr[-2000:] or r.stdout[-2000:]
        else:
            raise RuntimeError(f"codex exec failed: {err}")
    cache.write_text(json.dumps(answer))
    return answer, time.time() - t0
