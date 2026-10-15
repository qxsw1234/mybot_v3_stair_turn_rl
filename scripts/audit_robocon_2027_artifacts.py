#!/usr/bin/env python3
"""Write or verify the closeout evidence and local checkpoint SHA-256 manifest.

Uses only the standard library. Verification does not load pickle/Torch files.
Raw evidence and checkpoint contents remain local. Verification requires the
matching local data; a public clone contains only the inventory and summaries.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/2027_ROBOCON_ARTIFACT_MANIFEST.json"
CHECKPOINTS = (
    ("059950", "locomotion_baseline", "mybot_v3_sim2sim_phase3b_resume_059500/2026-10-03_07-04-56.647780", "059950"),
    ("064242", "preserved_original", "mybot_v3_low_bar_expert_resume_064043/2026-10-11_14-06-04.613002", "latest"),
    ("064341", "research_candidate_not_accepted", "mybot_v3_low_bar_expert_resume_064242/2026-10-12_03-07-21.669914", "latest"),
    ("064400", "research_backup_not_accepted", "mybot_v3_low_bar_expert_resume_064341/2026-10-12_03-45-02.595899", "064400"),
    ("064540", "rejected_cross_engine_regression", "mybot_v3_low_bar_expert_resume_064341/2026-10-12_03-45-02.595899", "latest"),
)


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def write_manifest(path):
    evidence = sorted(p for p in (ROOT / "evaluations/robocon_2027").rglob("*")
                      if p.is_file() and p.suffix in (".json", ".npz"))
    if not evidence:
        raise ValueError("No evidence found; refusing to create an empty manifest")
    for item in evidence:
        if item.suffix == ".json":
            json.loads(item.read_text())
    checkpoints = []
    for number, role, run, export in CHECKPOINTS:
        folder = ROOT / "runs" / run
        files = [folder / "parameters.pkl",
                 folder / "checkpoints" / ("ac_weights_" + number + ".pt"),
                 folder / "checkpoints" / ("body_" + export + ".jit"),
                 folder / "checkpoints" / ("adaptation_module_" + export + ".jit")]
        checkpoints.append({"checkpoint": number, "role": role,
                            "run": str(folder.relative_to(ROOT)),
                            "files": [file_record(item) for item in files]})
    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_at_capture": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip(),
        "source_note": "Capture includes uncommitted closeout changes. The containing Git commit defines the published source snapshot.",
        "acceptance_note": "Historical and diagnostic results are local evidence, not a passed stage gate. Consult the execution status and next-steps plan. Raw evidence and checkpoint binaries are not published; verification requires local data.",
        "stage2_open": False,
        "hardware_validated": False,
        "checkpoint_files_published": False,
        "raw_evidence_files_published": False,
        "checkpoints": checkpoints,
        "evidence": [file_record(item) for item in evidence],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Wrote {}: {} evidence files, {} checkpoint groups".format(
        path, len(evidence), len(checkpoints)))


def verify_manifest(path, include_checkpoints):
    manifest = json.loads(path.read_text())
    records = list(manifest["evidence"])
    if include_checkpoints:
        records.extend(item for ck in manifest["checkpoints"] for item in ck["files"])
    errors = []
    for expected in records:
        candidate = (ROOT / expected["path"]).resolve()
        if ROOT.resolve() not in candidate.parents or not candidate.is_file():
            errors.append("Missing or outside repository: " + expected["path"])
            continue
        actual = file_record(candidate)
        if any(actual[key] != expected[key] for key in ("bytes", "sha256")):
            errors.append("Changed: " + expected["path"])
    print(json.dumps({"verified": not errors, "files_checked": len(records),
                      "checkpoint_files_checked": include_checkpoints,
                      "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--include-checkpoints", action="store_true",
                        help="Also verify local runs/ weights, exports and configs")
    args = parser.parse_args()
    if args.write:
        write_manifest(args.manifest)
        return 0
    return verify_manifest(args.manifest, args.include_checkpoints)


if __name__ == "__main__":
    raise SystemExit(main())
