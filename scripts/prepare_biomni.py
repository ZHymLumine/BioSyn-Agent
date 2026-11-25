"""Fetch the compatible Biomni revision and apply BioSyn execution support."""

import argparse
import subprocess
from pathlib import Path

REVISION = "400c1f366b96a35ca253e13c9b06c5076af41d65"
REMOTE = "https://github.com/snap-stanford/Biomni.git"
ROOT = Path(__file__).resolve().parents[1]


def git(path, *args, check=True):
    return subprocess.run(["git", "-C", str(path), *args], check=check,
                          capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "Biomni")
    args = parser.parse_args()
    checkout = args.directory.resolve()
    if checkout.exists() and not (checkout / ".git").is_dir():
        raise RuntimeError(f"Expected an empty destination or Git checkout: {checkout}")
    if not checkout.exists():
        checkout.mkdir(parents=True)
        git(checkout, "init")
        git(checkout, "remote", "add", "origin", REMOTE)
        git(checkout, "fetch", "--depth", "1", "origin", REVISION)
        git(checkout, "checkout", "--detach", "FETCH_HEAD")
    actual = git(checkout, "rev-parse", "HEAD").stdout.strip()
    if actual != REVISION:
        raise RuntimeError(f"Expected Biomni {REVISION}; found {actual}. Choose a new directory.")
    patch = ROOT / "patches/biomni-execution.patch"
    expected = patch.read_text()
    current = git(checkout, "diff", "--", "biomni/agent/a1.py").stdout
    if current == expected:
        print(f"Biomni {REVISION} is ready at {checkout}")
        return
    if git(checkout, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise RuntimeError("The Biomni checkout contains other changes; choose a new directory.")
    git(checkout, "apply", "--check", str(patch))
    git(checkout, "apply", str(patch))
    print(f"Prepared Biomni {REVISION} at {checkout}")


if __name__ == "__main__":
    main()
