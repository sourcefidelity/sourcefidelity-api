"""Guards for sensitive operational terminology in publishable files."""

from pathlib import Path
import subprocess

import pytest


def test_tracked_files_avoid_sensitive_network_route_term():
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        pytest.skip("Git metadata is unavailable in this test environment")
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    prohibited = ("v" + "pn").encode()
    offending: list[str] = []
    for relative_bytes in tracked:
        if not relative_bytes:
            continue
        relative = relative_bytes.decode(errors="surrogateescape")
        path = root / relative
        try:
            if prohibited in path.name.lower().encode() or prohibited in path.read_bytes().lower():
                offending.append(relative)
        except (IsADirectoryError, OSError):
            continue
    assert not offending, f"Sensitive network-route terminology found in: {offending}"
