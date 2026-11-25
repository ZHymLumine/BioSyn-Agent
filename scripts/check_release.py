"""Check the repository file set before packaging or publication."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = {
    ".git", "Biomni", "OpenBioMed", "external", "data", "datasets", "resources",
    "models", "checkpoints", "runs", "results", "experiments", "artifacts", "logs",
    "reports", "build", "dist", "__pycache__", ".pytest_cache", ".ruff_cache", ".secrets",
}
WEIGHTS = {".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".hdf5", ".onnx"}


def generated(path):
    return any(p in GENERATED or p.startswith(".venv") or p.endswith(".egg-info")
               for p in path.parts)


def source_files():
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True)
    if git.returncode == 0 and Path(git.stdout.strip()).resolve() == ROOT:
        result = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                                check=True, capture_output=True, text=True)
        return [Path(p) for p in result.stdout.split("\0") if p]
    return [p.relative_to(ROOT) for p in ROOT.rglob("*")
            if p.is_file() and not generated(p.relative_to(ROOT))]


def main():
    issues = []
    files = source_files()
    if not files:
        raise SystemExit("No source files selected. Stage the release files first.")
    selected = set(files)
    required = {
        Path("README.md"), Path("pyproject.toml"), Path("src/biosyn/cli.py"),
        Path("configs/biomni/cpu.yaml"), Path("configs/biomni/local.yaml"),
        Path("tests/biomni/test_planner_execution.py"),
    }
    for missing in sorted(required - selected):
        issues.append(f"Required release file missing: {missing}")
    for relative in files:
        path = ROOT / relative
        if generated(relative) or path.suffix in WEIGHTS | {".jsonl", ".log", ".zip"}:
            issues.append(f"Generated or binary asset: {relative}")
        if path.name.startswith(".env") and path.name != ".env.example":
            issues.append(f"Credential configuration: {relative}")
        if path.is_symlink() or path.stat().st_size > 1_000_000:
            issues.append(f"Symlink or oversized file: {relative}")
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            issues.append(f"Binary file: {relative}")
            continue
        if re.search(r"(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----)", content):
            issues.append(f"Credential-like content: {relative}")
        if re.search(r"/(?:home/acd\w+|groups/\w+/acd\w+|Users/zym)/", content):
            issues.append(f"Machine-specific path: {relative}")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", content):
                if "://" in target or target.startswith("#"):
                    continue
                local = target.split("#", 1)[0]
                if local:
                    destination = (path.parent / local).resolve()
                    if not destination.exists():
                        issues.append(f"Broken documentation link in {relative}: {target}")
                    elif destination.is_file() and (
                        not destination.is_relative_to(ROOT)
                        or destination.relative_to(ROOT) not in selected
                    ):
                        issues.append(f"Documentation target excluded from release in {relative}: {target}")
    if issues:
        raise SystemExit("\n".join(issues))
    print(f"Checked {len(files)} source files and documentation links.")


if __name__ == "__main__":
    main()
