"""Ask Astra (gpt-6-astra) about images through the local Codex CLI.

Each call runs `codex exec` headless in a read-only sandbox, in an empty
temporary directory, with a JSON output schema. Answers are cached on disk by a
hash of model, reasoning effort, prompt, schema and image bytes, so a rerun
only pays for items whose inputs changed. Beside each answer, `<key>.usage.json`
keeps the seconds and token usage of the call that produced it.

The model defaults to Astra at medium effort; set DSA_ASTRA_MODEL and
DSA_ASTRA_EFFORT (or pass --model / --effort to a harness) to try another,
e.g. gpt-5.6-sol or gpt-5.6-luna.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

DEFAULT_MODEL, DEFAULT_EFFORT = "gpt-6-astra", "medium"
MODEL = os.environ.get("DSA_ASTRA_MODEL", DEFAULT_MODEL)
EFFORT = os.environ.get("DSA_ASTRA_EFFORT", DEFAULT_EFFORT)  # overrides the high default in ~/.codex/config.toml


def add_model_args(parser) -> None:
    """--model / --effort on a harness, defaulting to the env vars, else Astra medium."""
    parser.add_argument("--model", default=MODEL, help=f"Codex model (default {MODEL})")
    parser.add_argument("--effort", default=EFFORT, help=f"reasoning effort (default {EFFORT})")


def run_dir(out: Path, model: str, effort: str) -> Path:
    """Where a harness writes results: `out/<model>-<effort>`, so models never overwrite each other.

    Images and the answer cache stay in `out` and are shared (cache keys include the model)."""
    d = Path(out) / f"{model}-{effort}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def usage(cache_dir: str | Path, model: str, effort: str, keys: set[str] | None = None) -> dict:
    """Mean seconds and tokens per call over the recorded uncached calls of `model` at `effort`
    in a cache dir, restricted to `keys` (cache keys) when given."""
    rows = []
    for p in Path(cache_dir).glob("*.usage.json"):
        if p.name.startswith("._"):
            continue
        u = json.loads(p.read_text())
        if u["model"] == model and u["effort"] == effort and (keys is None or u["key"] in keys):
            rows.append(u)
    mean = lambda k: sum(u.get(k, 0) for u in rows) / len(rows) if rows else None
    return {"calls_recorded": len(rows), "sec_per_call": mean("sec"), "input_tokens": mean("input_tokens"),
            "cached_input_tokens": mean("cached_input_tokens"), "output_tokens": mean("output_tokens"),
            "reasoning_output_tokens": mean("reasoning_output_tokens")}


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
    if os.environ.get("DSA_ASTRA_CACHE_ONLY"):  # replay a past run without paying for misses
        raise RuntimeError(f"not cached: {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    t0, err = time.time(), None
    with tempfile.TemporaryDirectory() as tmp:
        schema_path, out = Path(tmp) / "schema.json", Path(tmp) / "out.json"
        schema_path.write_text(json.dumps(schema))
        cmd = ["codex", "exec", "--json", "-m", model, "-c", f"model_reasoning_effort={effort}",
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
                    tokens = {}
                    for line in r.stdout.splitlines():
                        if '"turn.completed"' in line:
                            tokens = json.loads(line).get("usage", {})
                    break
                except json.JSONDecodeError as e:
                    err = e
            else:
                err = r.stderr[-2000:] or r.stdout[-2000:]
        else:
            raise RuntimeError(f"codex exec failed: {err}")
    sec = time.time() - t0
    cache.write_text(json.dumps(answer))
    cache.with_suffix(".usage.json").write_text(json.dumps(
        {"key": cache.stem, "model": model, "effort": effort, "sec": sec, **tokens}))
    return answer, sec
