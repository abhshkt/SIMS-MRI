"""Shared helpers for SIMS-MRI experiment entrypoints.

These helpers are intentionally lightweight and script-oriented so the
existing Phase 1/2/3 entrypoints can share behavior without changing their
command-line interfaces or output layout.
"""

from __future__ import annotations

import csv
import fcntl
import os
import pathlib
import random
import re
import string
import time
from datetime import datetime
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

from sims_mri.loader import InferDataset, get_image_coordinate_grid_nib


def repo_root() -> pathlib.Path:
    """Return the repository root when running from the packaged src layout."""
    return pathlib.Path(__file__).resolve().parents[2]


def default_config_path(config_name: str = "oasis.yaml") -> str:
    """Return the canonical config path under configs/."""
    return str(repo_root() / "configs" / config_name)


def get_image_frame(dataset, config, batch_size: int = 5000, input_idx: int = 0):
    if input_idx == 0:  # groundtruth
        from multi_contrast_inr.dataset_utils import norm_grid

        resampled_path = str(dataset.lr_contrast1).replace(".nii.gz", "-resampled.nii.gz")
        if not os.path.exists(resampled_path):
            lr_image = sitk.ReadImage(str(dataset.lr_contrast1))
            desired_spacing = getattr(config.DATASET, "DESIRED_SPACING", [1.0, 1.0, 1.0])
            input_spacing = lr_image.GetSpacing()
            input_size = lr_image.GetSize()

            output_size = [
                int(round(input_size[i] * (input_spacing[i] / desired_spacing[i])))
                for i in range(len(input_size))
            ]

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(desired_spacing)
            resampler.SetSize(output_size)
            resampler.SetOutputOrigin(lr_image.GetOrigin())
            resampler.SetOutputDirection(lr_image.GetDirection())
            resampler.SetInterpolator(sitk.sitkLinear)
            resampled_image = resampler.Execute(lr_image)
            sitk.WriteImage(resampled_image, resampled_path)

        image_dict = get_image_coordinate_grid_nib(nib.load(resampled_path))
        mgrid = image_dict["coordinates"]
        mgrid_affine = image_dict["affine"]
        out_dim_xyz = image_dict["dim"]

        min_c, max_c = dataset.coordinates_minmax
        mgrid = norm_grid(mgrid, xmin=min_c, xmax=max_c)
    elif input_idx == 1:
        mgrid = dataset.get_coordinates1()
        mgrid_affine = dataset.get_affine1()
        out_dim_xyz = dataset.get_dim1()
    elif input_idx == 2:
        mgrid = dataset.get_coordinates2()
        mgrid_affine = dataset.get_affine2()
        out_dim_xyz = dataset.get_dim2()
    else:
        raise AssertionError("wrong index")

    infer_data = InferDataset(mgrid)
    infer_loader = torch.utils.data.DataLoader(
        infer_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.SETTINGS.NUM_WORKERS,
    )
    out_image = np.zeros((int(out_dim_xyz.prod()), 1))

    return (mgrid_affine, infer_loader, out_image, out_dim_xyz)


def generate_unique_id() -> str:
    """Generate unique experiment ID: timestamp_random_pid."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    pid = os.getpid()
    return f"{timestamp}_{random_str}_{pid}"


def hash_ckpt_for(model_path: str) -> str:
    """Map an image-model checkpoint path to the matching hash checkpoint path."""
    return model_path.replace("_model", "_hash")


def snapshot_state_to_cpu(model: Any):
    """Clone a module or tensor-like state dict payload to CPU."""
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def resolve_mlflow_tracking_uri(cli_uri: str | None = None) -> str:
    """Resolve MLflow tracking URI; defaults to a local file store in ./mlruns."""
    raw_uri = cli_uri if cli_uri is not None else os.environ.get("MLFLOW_TRACKING_URI")
    if raw_uri:
        raw_uri = str(raw_uri).strip()

    if not raw_uri:
        default_store = pathlib.Path("mlruns").resolve()
        return f"file:{default_store.as_posix()}"

    if raw_uri.startswith("file:") or "://" in raw_uri or raw_uri.startswith("databricks"):
        return raw_uri

    store_path = pathlib.Path(raw_uri).expanduser().resolve()
    return f"file:{store_path.as_posix()}"


def mlflow_log_metrics(mlflow_module, metrics, step=None) -> None:
    """Safely log numeric metrics to MLflow."""
    if mlflow_module is None:
        return

    clean_metrics = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            clean_metrics[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    if not clean_metrics:
        return

    mlflow_module.log_metrics(clean_metrics, step=step)


def safe_append_to_csv(csv_path: str, row_data, fieldnames) -> bool:
    """Safely append to CSV with file locking to handle concurrent writes."""
    max_retries = 10
    file_exists = os.path.exists(csv_path)

    for attempt in range(max_retries):
        try:
            with open(csv_path, "a", newline="") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    if not file_exists and handle.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row_data)
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                return True
        except IOError as exc:
            if attempt < max_retries - 1:
                time.sleep(0.1 + random.random() * 0.1)
            else:
                print(f"Failed to write to CSV after {max_retries} attempts: {exc}")
                return False
    return False


def shorten_project_name(project_name: str) -> str:
    """Shorten project names to keep paths concise."""
    if not project_name:
        return "proj"
    short = project_name
    replacements = {
        "rotation": "rot",
        "sigma": "s",
    }
    for old, new in replacements.items():
        short = short.replace(old, new)
    return short.replace("__", "_").strip("_")


def format_number_short(val) -> str:
    """Format numbers for filenames (no trailing zeros, '.' -> 'p')."""
    try:
        num = float(val)
        formatted = f"{num:g}"
    except (TypeError, ValueError):
        return str(val)
    return formatted.replace(".", "p")


def extract_parent_id_from_path(checkpoint_path: str) -> str | None:
    """Extract the parent experiment ID from a runs/{id}/... checkpoint path."""
    match = re.search(r"runs[/\\]([^/\\]+)[/\\]", checkpoint_path)
    if match:
        return match.group(1)
    return None
