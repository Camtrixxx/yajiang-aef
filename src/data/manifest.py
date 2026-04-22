from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class ManifestRecord:
    sample_id: str
    valid_start_ms: int
    valid_end_ms: int
    inputs: dict[str, dict[str, Any]]
    targets: dict[str, dict[str, Any]]
    split: str = "train"
    meta: dict[str, Any] | None = None


def _validate_input_source(source_name: str, payload: dict[str, Any]) -> None:
    if "frames" not in payload:
        raise ValueError(f"Manifest input source '{source_name}' missing field: frames")
    if not isinstance(payload["frames"], list):
        raise TypeError(f"Manifest input source '{source_name}'.frames must be list")
    for i, frame in enumerate(payload["frames"]):
        if "path" not in frame or "timestamp_ms" not in frame:
            raise ValueError(
                f"Manifest input source '{source_name}'.frames[{i}] "
                f"must contain path and timestamp_ms"
            )


def _validate_target_source(target_name: str, payload: dict[str, Any]) -> None:
    if "path" not in payload:
        raise ValueError(f"Manifest target '{target_name}' missing field: path")
    # relative_time 和 metadata 允许缺省
    if "relative_time" in payload and not isinstance(payload["relative_time"], (int, float)):
        raise TypeError(f"Manifest target '{target_name}'.relative_time must be number")
    if "metadata" in payload and not isinstance(payload["metadata"], list):
        raise TypeError(f"Manifest target '{target_name}'.metadata must be list[float]")


def validate_record(raw: dict[str, Any]) -> ManifestRecord:
    required = ["sample_id", "valid_start_ms", "valid_end_ms", "inputs", "targets"]
    for key in required:
        if key not in raw:
            raise ValueError(f"Manifest record missing required field: {key}")

    if not isinstance(raw["inputs"], dict):
        raise TypeError("Manifest record 'inputs' must be dict")
    if not isinstance(raw["targets"], dict):
        raise TypeError("Manifest record 'targets' must be dict")

    for src_name, payload in raw["inputs"].items():
        _validate_input_source(src_name, payload)

    for tgt_name, payload in raw["targets"].items():
        _validate_target_source(tgt_name, payload)

    return ManifestRecord(
        sample_id=str(raw["sample_id"]),
        valid_start_ms=int(raw["valid_start_ms"]),
        valid_end_ms=int(raw["valid_end_ms"]),
        inputs=raw["inputs"],
        targets=raw["targets"],
        split=str(raw.get("split", "train")),
        meta=raw.get("meta"),
    )


def load_manifest(path: str | Path, split: str | None = None) -> list[ManifestRecord]:
    path = Path(path)
    records: list[ManifestRecord] = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rec = validate_record(raw)
            if split is None or rec.split == split:
                records.append(rec)

    if len(records) == 0:
        split_info = f" for split='{split}'" if split is not None else ""
        raise ValueError(f"No manifest records found in {path}{split_info}")

    return records


def dump_manifest(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for raw in records:
            rec = validate_record(raw)
            f.write(
                json.dumps(
                    {
                        "sample_id": rec.sample_id,
                        "valid_start_ms": rec.valid_start_ms,
                        "valid_end_ms": rec.valid_end_ms,
                        "inputs": rec.inputs,
                        "targets": rec.targets,
                        "split": rec.split,
                        "meta": rec.meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
