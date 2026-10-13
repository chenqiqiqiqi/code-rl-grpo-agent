from __future__ import annotations

import urllib.request
from pathlib import Path

from posttrain.common import file_digest


def download_verified(url: str, destination: Path, expected_sha256: str | None = None) -> str:
    """Pinned data file only; never import/execute a dataset loading script."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        actual = file_digest(destination)
        if expected_sha256 and actual != expected_sha256:
            raise ValueError("existing_file_checksum_mismatch")
        return actual
    partial = destination.with_suffix(destination.suffix+'.partial')
    request = urllib.request.Request(url,headers={"User-Agent":"posttrain-project/0.1"})
    with urllib.request.urlopen(request,timeout=60) as response, partial.open('wb') as out:
        while block := response.read(1 << 20):
            out.write(block)
    actual = file_digest(partial)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError("download_checksum_mismatch")
    partial.replace(destination)
    return actual
