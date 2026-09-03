"""Prepare traceable 360-column HCP inputs for the prespecified evaluation.

The script supports two strict input routes:

1. Existing 360-column CIFTI parcel time series (`*.ptseries.nii`) or text arrays.
2. HCP `dtseries.nii` files parcellated with Connectome Workbench and an
   explicitly supplied 360-region parcellation file.

It deliberately does not support the earlier simplified route of taking the
first 360 columns from a 379-column file. That route is useful provenance for
older exploratory files, but it is not strict enough for a new prespecified
evaluation batch.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from shutil import which

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for import_root in (ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np

try:
    import nibabel as nib
except Exception:  # pragma: no cover - environment dependent
    nib = None

from run_prespecified_hcp_evaluation import (
    ALLOWED_PARCELLATION_METHODS,
    DEFAULT_PROVENANCE_DIR,
    PRESPECIFIED_SUBJECTS,
)


DATA_DIR = ROOT / "NPI-main" / "NPI-main" / "real_fMRI_data"
DOWNLOAD_ROOT = ROOT / "prespecified_hcp_downloads"
WORKBENCH_ENV = os.environ.get("CONNECTOME_WORKBENCH")
WORKBENCH_FALLBACK = Path(WORKBENCH_ENV) if WORKBENCH_ENV else None
DEFAULT_MMP_DLABEL = (
    ROOT
    / "data-all"
    / "HCP_S1200_Atlas"
    / "HCP_S1200_Atlas_Z4_pkXDZ"
    / "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
)
SESSIONS = ["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"]
DT_SERIES_BASENAMES = [
    "rfMRI_{session}_Atlas_MSMAll_hp2000_clean.dtseries.nii",
    "rfMRI_{session}_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii",
]


def hcp_s3_url(subject: str, session: str) -> str:
    return f"https://hcp-openaccess.s3.amazonaws.com/{hcp_s3_key(subject, session)}"


def hcp_s3_key(subject: str, session: str) -> str:
    return (
        f"HCP_1200/{subject}/"
        f"MNINonLinear/Results/rfMRI_{session}/"
        f"rfMRI_{session}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
    )


def find_workbench() -> Path | None:
    path = which("wb_command")
    if path:
        return Path(path)
    if WORKBENCH_FALLBACK is not None and WORKBENCH_FALLBACK.exists():
        return WORKBENCH_FALLBACK
    return None


def aws_credentials_file_status() -> dict:
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    credentials_file = home / ".aws" / "credentials"
    status = {
        "aws_credentials_file": str(credentials_file),
        "aws_credentials_file_exists": credentials_file.exists(),
        "aws_credentials_default_profile": False,
        "aws_credentials_file_has_access_key_id": False,
        "aws_credentials_file_has_secret_access_key": False,
        "aws_credentials_file_has_session_token": False,
    }
    if not credentials_file.exists():
        return status
    parser = configparser.ConfigParser()
    try:
        parser.read(credentials_file, encoding="utf-8")
    except Exception:
        return status
    if not parser.has_section("default"):
        return status
    profile = parser["default"]
    status.update(
        {
            "aws_credentials_default_profile": True,
            "aws_credentials_file_has_access_key_id": bool(str(profile.get("aws_access_key_id", "")).strip()),
            "aws_credentials_file_has_secret_access_key": bool(str(profile.get("aws_secret_access_key", "")).strip()),
            "aws_credentials_file_has_session_token": bool(str(profile.get("aws_session_token", "")).strip()),
        }
    )
    return status


def audit_environment() -> dict:
    wb = find_workbench()
    aws_cli = which("aws") is not None
    curl_cli = which("curl") is not None or which("curl.exe") is not None
    credential_names = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "HCP_USERNAME",
        "HCP_PASSWORD",
        "CONNECTOMEDB_USER",
        "CONNECTOMEDB_PASSWORD",
    ]
    set_credentials = [
        name for name in credential_names if os.environ.get(name)
    ]
    file_credentials = aws_credentials_file_status()
    has_file_credentials = (
        file_credentials["aws_credentials_default_profile"]
        and file_credentials["aws_credentials_file_has_access_key_id"]
        and file_credentials["aws_credentials_file_has_secret_access_key"]
    )
    return {
        "aws_cli": bool(aws_cli),
        "curl": bool(curl_cli),
        "wb_command": wb is not None,
        "wb_command_path": str(wb) if wb else None,
        "default_mmp_dlabel": str(DEFAULT_MMP_DLABEL),
        "default_mmp_dlabel_exists": DEFAULT_MMP_DLABEL.exists(),
        **file_credentials,
        "has_hcp_or_aws_credentials": bool(set_credentials) or bool(has_file_credentials),
        "set_credential_names": set_credentials,
    }


def resolve_parcellation_file(parcellation_file: Path | None = None) -> Path:
    path = Path(parcellation_file) if parcellation_file is not None else DEFAULT_MMP_DLABEL
    if not path.exists():
        raise FileNotFoundError(
            "Missing 360-region parcellation file. Provide --parcellation_file "
            f"or place the default HCP-MMP1 dlabel at {DEFAULT_MMP_DLABEL}"
        )
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_array(path: Path) -> np.ndarray:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        array = np.load(str(path))
    elif suffixes.endswith(".txt") or suffixes.endswith(".tsv") or suffixes.endswith(".csv"):
        array = np.loadtxt(str(path))
    elif suffixes.endswith(".nii") and nib is not None:
        array = np.asarray(nib.load(str(path)).get_fdata())
    else:
        raise ValueError(f"Unsupported input type or missing nibabel: {path}")
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D time-by-region array, got {array.shape} for {path}")
    if array.shape[1] == 360:
        return array.astype(np.float32)
    if array.shape[0] == 360:
        return array.T.astype(np.float32)
    raise ValueError(f"Input must have exactly 360 columns or rows, got {array.shape} for {path}")


def merge_session_arrays(sessions: dict[str, np.ndarray]) -> np.ndarray:
    missing = [session for session in SESSIONS if session not in sessions]
    if missing:
        raise ValueError(f"Missing required sessions: {missing}")
    arrays = []
    for session in SESSIONS:
        array = np.asarray(sessions[session])
        if array.ndim != 2 or array.shape[1] != 360:
            raise ValueError(f"{session} must have exactly 360 columns, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{session} contains NaN or Inf values")
        arrays.append(array.astype(np.float32))
    merged = np.concatenate(arrays, axis=0)
    if merged.shape[0] <= 70:
        raise ValueError(f"Merged input has too few timepoints: {merged.shape}")
    return merged


def parcellate_dtseries_with_workbench(
    dtseries_file: Path,
    parcellation_file: Path,
    output_ptseries: Path,
    wb_command: Path | None = None,
) -> Path:
    wb = Path(wb_command) if wb_command else find_workbench()
    if wb is None or not wb.exists():
        raise FileNotFoundError("Connectome Workbench wb_command was not found")
    dtseries_file = Path(dtseries_file)
    parcellation_file = Path(parcellation_file)
    output_ptseries = Path(output_ptseries)
    if not dtseries_file.exists():
        raise FileNotFoundError(f"Missing dtseries file: {dtseries_file}")
    if not parcellation_file.exists():
        raise FileNotFoundError(f"Missing parcellation file: {parcellation_file}")
    output_ptseries.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(wb),
        "-cifti-parcellate",
        str(dtseries_file),
        str(parcellation_file),
        "COLUMN",
        str(output_ptseries),
    ]
    completed = subprocess.run(command, cwd=str(ROOT), check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "wb_command -cifti-parcellate failed: "
            + (completed.stderr or completed.stdout or "").strip()
        )
    return output_ptseries


def write_model_input(subject: str, session_files: dict[str, Path], output_dir: Path = DATA_DIR) -> Path:
    arrays = {session: load_array(path) for session, path in session_files.items()}
    merged = merge_session_arrays(arrays)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"sub-{subject}_bold_360.npy"
    np.save(str(output), merged)
    return output


def write_provenance(
    subject: str,
    model_input: Path,
    source_files: list[Path],
    processing_scripts: list[str],
    parcellation_method: str,
    provenance_dir: Path = DEFAULT_PROVENANCE_DIR,
) -> Path:
    if parcellation_method not in ALLOWED_PARCELLATION_METHODS:
        raise ValueError(
            "parcellation_method must be one of "
            f"{sorted(ALLOWED_PARCELLATION_METHODS)}, got {parcellation_method!r}"
        )
    model_input = Path(model_input)
    if not model_input.exists():
        raise FileNotFoundError(f"Missing model input: {model_input}")
    normalized_sources = [Path(path) for path in source_files]
    missing_sources = [str(path) for path in normalized_sources if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(f"Missing provenance source files: {missing_sources}")
    provenance_dir = Path(provenance_dir)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "subject": str(subject),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_input": str(model_input),
        "model_input_sha256": sha256_file(model_input),
        "source_files": [str(path) for path in normalized_sources],
        "source_file_sha256": {str(path): sha256_file(path) for path in normalized_sources},
        "processing_scripts": list(processing_scripts),
        "parcellation_method": parcellation_method,
        "notes": (
            "Prepared for the prespecified traceable HCP evaluation. "
            "No 379-to-first-360-column truncation route was used."
        ),
    }
    output = provenance_dir / f"{subject}_provenance.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def expected_download_paths(subject: str, download_root: Path = DOWNLOAD_ROOT) -> dict[str, Path]:
    base = Path(download_root) / subject
    return {
        session: base / DT_SERIES_BASENAMES[0].format(session=session)
        for session in SESSIONS
    }


def candidate_hcp_dtseries_paths(subject: str, download_root: Path = DOWNLOAD_ROOT) -> dict[str, list[Path]]:
    flat_base = Path(download_root) / subject
    base = Path(download_root) / subject / "MNINonLinear" / "Results"
    return {
        session: [
            flat_base / template.format(session=session)
            for template in DT_SERIES_BASENAMES
        ]
        + [
            base / f"rfMRI_{session}" / template.format(session=session)
            for template in DT_SERIES_BASENAMES
        ]
        for session in SESSIONS
    }


def native_hcp_dtseries_paths(subject: str, download_root: Path = DOWNLOAD_ROOT) -> dict[str, Path]:
    candidates = candidate_hcp_dtseries_paths(subject, download_root)
    return {session: candidates[session][2] for session in SESSIONS}


def find_dtseries_paths(subject: str, download_root: Path = DOWNLOAD_ROOT) -> dict[str, Path]:
    candidates = candidate_hcp_dtseries_paths(subject, download_root)
    resolved = {}
    for session in SESSIONS:
        resolved[session] = next(
            (path for path in candidates[session] if path.exists()),
            candidates[session][0],
        )
    return resolved


def download_hcp_dtseries_with_curl(subject: str, session: str, output_file: Path) -> Path:
    """Download one HCP dtseries file with curl.

    This works only if the HCP/S3 endpoint is accessible from the current
    environment and the user's access terms/credentials permit the request.
    """
    curl = which("curl.exe") or which("curl")
    if not curl:
        raise FileNotFoundError("curl was not found")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        curl,
        "-L",
        "--fail",
        "--retry",
        "3",
        "--connect-timeout",
        "20",
        "--max-time",
        "120",
        "-o",
        str(output_file),
        hcp_s3_url(subject, session),
    ]
    completed = subprocess.run(command, cwd=str(ROOT), check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        if output_file.exists():
            output_file.unlink()
        raise RuntimeError((completed.stderr or completed.stdout or "curl download failed").strip())
    return output_file


def download_hcp_dtseries_with_s3(
    subject: str,
    session: str,
    output_file: Path,
    client=None,
    chunk_size: int = 8 * 1024 * 1024,
    max_retries: int = 5,
) -> Path:
    """Download one HCP dtseries file with signed S3 credentials."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if client is None:
        try:
            import botocore.session
            from botocore.config import Config
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("botocore is required for signed HCP S3 downloads") from exc
        client = botocore.session.get_session().create_client(
            "s3",
            region_name="us-east-1",
            config=Config(
                signature_version="s3v4",
                connect_timeout=30,
                read_timeout=60,
                retries={"max_attempts": 10, "mode": "standard"},
            ),
        )

    key = hcp_s3_key(subject, session)
    head = client.head_object(Bucket="hcp-openaccess", Key=key)
    expected_size = int(head["ContentLength"])
    if output_file.exists() and output_file.stat().st_size == expected_size:
        return output_file
    if output_file.exists() and output_file.stat().st_size != expected_size:
        output_file.unlink()

    part_file = output_file.with_suffix(output_file.suffix + ".part")
    start = part_file.stat().st_size if part_file.exists() else 0
    if start > expected_size:
        part_file.unlink()
        start = 0
    try:
        with part_file.open("ab") as handle:
            while start < expected_size:
                end = min(start + chunk_size - 1, expected_size - 1)
                byte_range = f"bytes={start}-{end}"
                last_error = None
                for attempt in range(max_retries + 1):
                    try:
                        response = client.get_object(Bucket="hcp-openaccess", Key=key, Range=byte_range)
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt >= max_retries:
                            raise
                        time.sleep(min(2**attempt, 30))
                else:  # pragma: no cover - defensive
                    raise last_error or RuntimeError(f"Failed to read {byte_range}")
                written = 0
                for chunk in response["Body"].iter_chunks(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)
                expected_written = end - start + 1
                if written != expected_written:
                    raise IOError(
                        f"Short S3 range read for {key} {byte_range}: "
                        f"expected {expected_written} bytes, got {written}"
                    )
                start = end + 1
                handle.flush()
    except Exception:
        raise
    if part_file.stat().st_size != expected_size:
        raise IOError(
            f"Incomplete S3 download for {key}: expected {expected_size} bytes, "
            f"got {part_file.stat().st_size}"
        )
    part_file.replace(output_file)
    return output_file


def download_hcp_dtseries(subject: str, session: str, output_file: Path) -> Path:
    if audit_environment().get("has_hcp_or_aws_credentials"):
        return download_hcp_dtseries_with_s3(subject, session, output_file)
    return download_hcp_dtseries_with_curl(subject, session, output_file)


def process_subject_from_dtseries(
    subject: str,
    download_root: Path = DOWNLOAD_ROOT,
    output_dir: Path = DATA_DIR,
    provenance_dir: Path = DEFAULT_PROVENANCE_DIR,
    parcellation_file: Path | None = None,
    wb_command: Path | None = None,
) -> dict:
    """Convert four HCP dtseries sessions into one strict 360-column input."""
    subject = str(subject)
    parcellation_path = resolve_parcellation_file(parcellation_file)
    wb = Path(wb_command) if wb_command is not None else find_workbench()
    if wb is None or not wb.exists():
        raise FileNotFoundError("Connectome Workbench wb_command was not found")

    dtseries_files = find_dtseries_paths(subject, download_root)
    missing = [str(path) for path in dtseries_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required HCP dtseries files: {missing}")

    parcellated_dir = Path(download_root) / subject / "parcellated"
    session_ptseries: dict[str, Path] = {}
    for session in SESSIONS:
        output_ptseries = parcellated_dir / f"rfMRI_{session}_MMP360.ptseries.nii"
        session_ptseries[session] = parcellate_dtseries_with_workbench(
            dtseries_files[session],
            parcellation_path,
            output_ptseries,
            wb_command=wb,
        )

    model_input = write_model_input(subject, session_ptseries, output_dir=output_dir)
    provenance = write_provenance(
        subject=subject,
        model_input=model_input,
        source_files=[*dtseries_files.values(), parcellation_path],
        processing_scripts=[
            str(ROOT / "prepare_prespecified_hcp_inputs.py"),
            f"{wb} -cifti-parcellate",
        ],
        parcellation_method="official_hcp_mmp1_360",
        provenance_dir=provenance_dir,
    )
    return {
        "subject": subject,
        "status": "processed",
        "model_input": str(model_input),
        "provenance": str(provenance),
        "source_dtseries": {session: str(path) for session, path in dtseries_files.items()},
        "parcellated_session_files": {session: str(path) for session, path in session_ptseries.items()},
        "parcellation_file": str(parcellation_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_environment", action="store_true")
    parser.add_argument("--subjects", nargs="+", default=PRESPECIFIED_SUBJECTS)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--process_existing", action="store_true")
    parser.add_argument("--download_root", type=Path, default=DOWNLOAD_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--provenance_dir", type=Path, default=DEFAULT_PROVENANCE_DIR)
    parser.add_argument("--parcellation_file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_environment:
        print(json.dumps(audit_environment(), indent=2))
        return
    if args.download:
        records = []
        for subject in args.subjects:
            for session, path in expected_download_paths(subject, args.download_root).items():
                try:
                    downloaded = download_hcp_dtseries(subject, session, path)
                    records.append({"subject": subject, "session": session, "status": "downloaded", "path": str(downloaded)})
                except Exception as exc:
                    records.append({"subject": subject, "session": session, "status": "failed", "error": str(exc)})
                    break
        print(json.dumps(records, indent=2))
        return
    if args.process_existing:
        records = []
        for subject in args.subjects:
            try:
                records.append(
                    process_subject_from_dtseries(
                        subject,
                        download_root=args.download_root,
                        output_dir=args.output_dir,
                        provenance_dir=args.provenance_dir,
                        parcellation_file=args.parcellation_file,
                    )
                )
            except Exception as exc:
                records.append({"subject": str(subject), "status": "failed", "error": str(exc)})
        print(json.dumps(records, indent=2))
        return
    print(json.dumps(audit_environment(), indent=2))


if __name__ == "__main__":
    main()
