"""Bundle public TrashNet + project code for the Phase 3 Colab notebook."""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.dataset import ROOT
from src.prepare_data import EXPECTED_ARCHIVE_SHA256, sha256_file

OUTPUT = ROOT / "artifacts/phase3_colab_bundle.zip"


def main() -> None:
    archive = ROOT / "data/raw/trashnet-dataset-resized.zip"
    if sha256_file(archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("TrashNet archive checksum mismatch")
    paths = sorted((ROOT / "src").glob("*.py"))
    paths += sorted((ROOT / "configs").glob("*.yaml"))
    paths += [ROOT / "data/processed/trashnet_manifest.csv", archive]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("Source, config, manifest or archive missing")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w") as bundle:
        for path in paths:
            compression = zipfile.ZIP_STORED if path == archive else zipfile.ZIP_DEFLATED
            bundle.write(path, path.relative_to(ROOT), compress_type=compression)
    digest = sha256_file(OUTPUT)
    print(f"bundle={OUTPUT}")
    print(f"size_bytes={OUTPUT.stat().st_size}")
    print(f"sha256={digest}")
    print("Contents:")
    for path in paths:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
