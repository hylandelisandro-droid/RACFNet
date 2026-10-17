"""Install the RACFNet-original overlay into an official QSFNet checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


OVERLAY_DIRS = ("racfnet", "round2", "round3", "scripts", "evaluation", "configs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qsfnet", required=True, type=Path, help="Official QSFNet checkout")
    parser.add_argument(
        "--racfnet",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="This RACFNet repository",
    )
    args = parser.parse_args()

    qsfnet = args.qsfnet.resolve()
    racfnet = args.racfnet.resolve()
    required = (qsfnet / "Tnet.py", qsfnet / "QAnet.py", qsfnet / "lib")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Not an official QSFNet checkout; missing: " + ", ".join(missing))

    integrated = qsfnet / "round2" / "RACFNet.py"
    if integrated.exists():
        text = integrated.read_text(encoding="utf-8", errors="ignore")
        if "from racfnet.modules import CCCScale, RTFScale" in text:
            print("RACFNet overlay is already installed; no changes made.")
            return
        raise FileExistsError(f"Refusing to overwrite existing file: {integrated}")

    for name in OVERLAY_DIRS:
        source = racfnet / name
        if source.exists():
            shutil.copytree(source, qsfnet / name, dirs_exist_ok=True)
    shutil.copy2(racfnet / "requirements.txt", qsfnet / "requirements.racfnet.txt")
    integrated.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(qsfnet / "Tnet.py", integrated)
    normalized = integrated.read_text(encoding="utf-8").replace("\r\n", "\n")
    integrated.write_text(normalized, encoding="utf-8", newline="\n")

    patch = racfnet / "patches" / "qsfnet_to_racfnet.patch"
    check = subprocess.run(
        ["git", "apply", "--check", "--whitespace=nowarn", str(patch)],
        cwd=qsfnet,
        text=True,
        capture_output=True,
    )
    if check.returncode != 0:
        integrated.unlink(missing_ok=True)
        raise RuntimeError("Patch compatibility check failed:\n" + check.stderr)
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch)],
        cwd=qsfnet,
        check=True,
    )
    print(f"RACFNet overlay installed into: {qsfnet}")


if __name__ == "__main__":
    main()
