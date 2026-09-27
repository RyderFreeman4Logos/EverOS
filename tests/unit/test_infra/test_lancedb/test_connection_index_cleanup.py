"""Indexed readers must not retain cleaned-up vector-index files."""

from __future__ import annotations

import os
import random
from datetime import timedelta
from pathlib import Path

from lancedb.index import IvfFlat

from everos.config import LanceDBSettings
from everos.core.persistence import open_lancedb_connection


def _deleted_index_fds(root: Path) -> list[str]:
    targets = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if str(root) in target and target.endswith(" (deleted)") and ".idx" in target:
            targets.append(target)
    return targets


async def test_search_releases_cleaned_vector_files(tmp_path: Path) -> None:
    conn = await open_lancedb_connection(tmp_path / "vectors", LanceDBSettings())
    rng = random.Random(1)
    rows = [
        {"id": str(i), "vector": [rng.random() for _ in range(16)]} for i in range(600)
    ]
    table = await conn.create_table("vectors", data=rows)
    config = IvfFlat(distance_type="cosine", num_partitions=4)
    try:
        await table.create_index("vector", config=config)
        for _ in range(3):
            result = await (
                table.query()
                .nearest_to([0.1] * 16)
                .column("vector")
                .nprobes(4)
                .limit(5)
                .to_list()
            )
            assert len(result) == 5
            await table.create_index("vector", replace=True, config=config)
            await table.optimize(cleanup_older_than=timedelta(seconds=0))
            assert not _deleted_index_fds(tmp_path / "vectors")
    finally:
        table.close()
        conn.close()
