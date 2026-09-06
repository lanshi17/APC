from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Dataset:
    """JSONL 数据集：dev / validation / holdout / perturbation，带版本指纹。"""

    dataset_id: str
    version: str
    samples: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def sample_id(self, index: int) -> str:
        return f"{self.dataset_id}#{index:04d}"


def load_dataset(path: str | Path, dataset_id: str | None = None) -> Dataset:
    path = Path(path)
    samples = []
    content_hash = hashlib.sha256()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            content_hash.update(line.encode())
    return Dataset(
        dataset_id=dataset_id or path.stem,
        version=content_hash.hexdigest()[:12],
        samples=samples,
    )
