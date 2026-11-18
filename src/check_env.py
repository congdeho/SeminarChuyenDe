"""In môi trường thực tế, không cài hoặc tải thêm bất cứ thứ gì."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from .labels import LABELS, LABELS_VI


PACKAGES = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("opencv-python", "cv2"),
    ("numpy", "numpy"),
    ("pillow", "PIL"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("pyyaml", "yaml"),
    ("tqdm", "tqdm"),
)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".cache" / "matplotlib"))
    result: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "project_root": str(project_root),
        "labels": list(LABELS),
        "labels_vi": list(LABELS_VI),
        "packages": {},
        "devices": {"cpu": False, "mps_built": False, "mps": False, "cuda": False},
    }

    packages = result["packages"]
    assert isinstance(packages, dict)
    missing: list[str] = []
    for distribution, module_name in PACKAGES:
        try:
            packages[distribution] = importlib.metadata.version(distribution)
            importlib.import_module(module_name)
        except (importlib.metadata.PackageNotFoundError, ImportError, OSError) as error:
            packages[distribution] = f"missing or failed to import: {error}"
            missing.append(distribution)

    if "torch" not in missing:
        torch = importlib.import_module("torch")
        devices = result["devices"]
        assert isinstance(devices, dict)
        devices["cpu"] = bool(torch.ones(2).sum().item() == 2)
        devices["mps_built"] = bool(torch.backends.mps.is_built())
        devices["mps"] = bool(torch.backends.mps.is_available())
        devices["cuda"] = bool(torch.cuda.is_available())
        if devices["mps"]:
            devices["mps_smoke_test"] = bool(torch.ones(2, device="mps").sum().item() == 2)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if missing:
        print("Missing or unusable packages: " + ", ".join(missing), file=sys.stderr)
        return 1
    devices = result["devices"]
    assert isinstance(devices, dict)
    if not devices["cpu"]:
        print("PyTorch CPU smoke test failed", file=sys.stderr)
        return 1
    if not (sys.version_info.major == 3 and sys.version_info.minor == 12):
        print("Project requires Python 3.12", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
