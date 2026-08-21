from __future__ import annotations

"""Check that every repository path cited by the documentation is real.

The interview documents lean heavily on "here is the file that proves it".
That argument collapses if a reviewer clones the repository and the cited path
does not exist, or exists only on the author's machine because it is ignored by
git. This validator therefore enforces two things per referenced path:

  1. it exists in the working tree, and
  2. it is visible to a fresh clone (not git-ignored).

Paths that are deliberately not redistributed -- third-party source video and
model weights -- are declared in ``NOT_REDISTRIBUTED``. They are allowed to be
ignored, but they must still be described as such by the documentation, so the
validator requires the citing document to say where they come from rather than
silently pointing at a missing file.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

# Path-like tokens inside backticks, e.g. `docs/TRAINING.md`, `outputs/x.json`.
# Restricted to the project's real top-level directories so that prose such as
# `beta=0.1` or a remote absolute path is not mistaken for a repository file.
TRACKED_ROOTS = ("src/", "docs/", "scripts/", "configs/", "data/", "tests/", "outputs/")
PATH_PATTERN = re.compile(r"`([A-Za-z0-9_./-]+)`")
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)#]+)\)")

# These exist locally but are intentionally absent from a clone. Each entry maps
# to a substring that the citing document must contain, so the reader is told
# how to obtain the file instead of hitting a dead reference.
VIDEO_POINTERS = ("data/raw/README.md", "docs/USAGE.md")
WEIGHT_POINTERS = ("docs/REVALIDATION.md", "docs/USAGE.md", "docs/TRAINING.md")
NOT_REDISTRIBUTED = {
    "data/raw/cola_review.mp4": VIDEO_POINTERS,
    "data/raw/safedroid_demo.mp4": VIDEO_POINTERS,
    "data/raw/yoga.mp4": VIDEO_POINTERS,
    "outputs/models/neural_reranker.pt": WEIGHT_POINTERS,
    "outputs/models/answer_verifier.pkl": WEIGHT_POINTERS,
}

# Directories the service creates at runtime. Documentation legitimately names
# them as destinations, so they are exempt from the "must exist" rule -- but
# only these, and only as directories.
RUNTIME_PATHS = frozenset({"data/uploads", "outputs/runs", "outputs/indexes"})


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-validate-documentation-links")
    parser.add_argument("--output", default="outputs/reports/documentation_links.json")
    args = parser.parse_args()
    report = validate_documentation_links(ROOT)
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    if ROOT.resolve() not in (output, *output.parents):
        raise SystemExit("documentation link report must remain inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" keeps the digest of this tracked report identical on
    # Windows and Linux; the delivery validator hashes these bytes.
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


def validate_documentation_links(root: Path) -> dict:
    root = root.resolve()
    documents = sorted([root / "README.md", *(root / "docs").glob("*.md")])
    ignored = _ignored_paths(root)

    failures: list[dict] = []
    checked = 0
    for document in documents:
        if not document.is_file():
            continue
        text = document.read_text(encoding="utf-8")
        relative_doc = document.relative_to(root).as_posix()
        for reference in _references(text):
            checked += 1
            failure = _check_reference(root, relative_doc, reference, text, ignored)
            if failure is not None:
                failures.append(failure)

    return {
        "schema_version": "videotrace-documentation-links-v1",
        "valid": not failures,
        "documents": len(documents),
        "references_checked": checked,
        "failures": failures,
    }


def _check_reference(
    root: Path,
    document: str,
    reference: str,
    text: str,
    ignored: frozenset[str],
) -> dict | None:
    target = root / reference
    if reference in RUNTIME_PATHS:
        return None

    pointers = NOT_REDISTRIBUTED.get(reference)
    if pointers is not None:
        # These are absent from a clone by design and may also be absent from
        # the author's tree. Either way the requirement is the same: the citing
        # document must send the reader somewhere that explains how to obtain
        # the file -- or be that explanation itself.
        if document in pointers or any(pointer in text for pointer in pointers):
            return None
        return {
            "document": document,
            "reference": reference,
            "reason": "not redistributed; document must point the reader at one of "
            + ", ".join(pointers),
        }

    if not target.exists():
        return {"document": document, "reference": reference, "reason": "missing"}

    if reference not in ignored:
        return None

    if target.is_dir():
        # A directory counts as visible when it contributes at least one file to
        # a clone -- e.g. data/raw ships only its README, which is the point.
        if any(
            child.relative_to(root).as_posix() not in ignored
            for child in target.rglob("*")
            if child.is_file()
        ):
            return None

    return {
        "document": document,
        "reference": reference,
        "reason": "git-ignored; invisible to a fresh clone",
    }


def _references(text: str) -> list[str]:
    found: list[str] = []
    for match in (*PATH_PATTERN.findall(text), *MARKDOWN_LINK_PATTERN.findall(text)):
        candidate = match.strip().lstrip("./")
        if candidate.startswith(TRACKED_ROOTS) or candidate in {"README.md", "LICENSE"}:
            if "*" not in candidate and candidate not in found:
                found.append(candidate)
    return found


def _ignored_paths(root: Path) -> frozenset[str]:
    """Return the set of working-tree paths git would exclude from a clone."""

    result = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--directory"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # Without git (e.g. a source tarball) the visibility half of the check
        # cannot be evaluated; existence is still enforced above.
        return frozenset()
    entries = [line.strip().rstrip("/") for line in result.stdout.splitlines() if line.strip()]
    ignored = set(entries)
    for entry in entries:
        directory = root / entry
        if directory.is_dir():
            ignored.update(
                child.relative_to(root).as_posix()
                for child in directory.rglob("*")
                if child.is_file()
            )
    return frozenset(ignored)


if __name__ == "__main__":
    main()
