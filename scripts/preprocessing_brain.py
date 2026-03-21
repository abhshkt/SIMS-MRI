"""Preprocess OASIS-3 brain MRI scans for SIMS-MRI training."""

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import nibabel.processing as nip
import numpy as np
import pandas as pd

from sims_mri.utils import get_rotation_matrix_numpy

RESAMPLING_FACTORS = {
    "axial": (1, 1, 4),
    "coronal": (1, 4, 1),
    "sagittal": (4, 1, 1),
}


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3):
    """Resample a NIfTI image to the given voxel spacing."""
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    new_shp = tuple(
        np.rint(
            [
                shp[0] * zms[0] / voxel_spacing[0],
                shp[1] * zms[1] / voxel_spacing[1],
                shp[2] * zms[2] / voxel_spacing[2],
            ]
        ).astype(int)
    )
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=0)
    print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img


def resample_and_save(input_file, output_file, view):
    """Load a NIfTI, resample to the given view's factor, and save."""
    img = nib.load(input_file)
    voxel_spacing = RESAMPLING_FACTORS.get(view.lower())
    if voxel_spacing is None:
        print(f"Invalid view: {view}. Use axial, sagittal, or coronal.")
        return
    new_img = resample_nib(img, voxel_spacing=voxel_spacing)
    nib.save(new_img, output_file)
    print(f"Resampled image saved as {output_file}")


def preprocess_oasis_subject(input_fname, subject_id, output_root=Path("oasis_random")):
    subject_output_dir = output_root / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(input_fname).name
    output_base = subject_output_dir / fname
    print(f"[*] Preprocessing {fname}")
    shutil.copyfile(input_fname, output_base)

    # Axial and coronal LR views
    for view in ["axial", "coronal"]:
        output_fname = str(output_base).replace(".nii.gz", f"_{view}_LR.nii.gz")
        resample_and_save(input_file=str(output_base), output_file=output_fname, view=view)

    # Padded baseline (for CNN based baseline)
    padded_target_shape = (256, 256, 256)
    img = nib.load(output_base)
    data = np.asanyarray(img.dataobj)
    affine = img.affine
    padded = np.zeros(padded_target_shape, dtype=data.dtype)
    src_shape = data.shape

    slices = []
    for axis in range(3):
        target_dim = padded_target_shape[axis]
        src_dim = src_shape[axis]
        if src_dim > target_dim:
            raise ValueError(
                f"Source dimension {src_dim} exceeds target {target_dim} along axis {axis} "
                f"for volume {fname}"
            )
        pad_before = (target_dim - src_dim) // 2
        pad_after = target_dim - src_dim - pad_before
        slices.append(slice(pad_before, target_dim - pad_after))

    padded[tuple(slices)] = data
    padded_fname = str(output_base).replace(".nii.gz", "_padded.nii.gz")
    nib.save(nib.Nifti1Image(padded, affine=affine), padded_fname)

    for view in ["axial", "sagittal", "coronal"]:
        output_fname = padded_fname.replace(".nii.gz", f"_{view}_LR.nii.gz")
        resample_and_save(input_file=padded_fname, output_file=output_fname, view=view)

    # Per-subject RNG for reproducible random transforms
    img = nib.load(output_base)
    rng = np.random.RandomState(hash(subject_id) % (2**31))
    base = str(output_base)

    # Translation (affine shift)
    translate_vec = np.append(rng.uniform(-8, 8, size=3), 0)
    print(f"[*] Random translation for {subject_id}: {translate_vec[:3]} mm")
    new_affine = img.affine.copy()
    new_affine[:, -1] += translate_vec
    nib.save(
        nib.Nifti1Image(np.asanyarray(img.dataobj), affine=new_affine),
        base.replace(".nii.gz", "_translated.nii.gz"),
    )

    # Translation (voxel shift, affine unchanged)
    orig_data = np.array(img.dataobj)
    shifted = np.zeros(orig_data.shape)
    shifted[8:, 8:, 8:] = orig_data[:-8, :-8, :-8]
    nib.save(
        nib.Nifti1Image(shifted, affine=img.affine),
        base.replace(".nii.gz", "_translated2.nii.gz"),
    )

    # Rotation only
    theta_x = rng.uniform(-0.1, 0.1)
    print(f"[*] Random x-axis rotation for {subject_id}: {theta_x:.6f} rad")
    rotation_mat = get_rotation_matrix_numpy(theta_x, 0.0, 0.0)
    rotated_affine = img.affine.copy().dot(rotation_mat)
    nib.save(
        nib.Nifti1Image(np.asanyarray(img.dataobj), affine=rotated_affine),
        base.replace(".nii.gz", "_rotated.nii.gz"),
    )

    # Rigid (rotation + translation)
    rigid_affine = img.affine.copy().dot(rotation_mat)
    rigid_affine[:, -1] += translate_vec
    nib.save(
        nib.Nifti1Image(np.asanyarray(img.dataobj), affine=rigid_affine),
        base.replace(".nii.gz", "_rigid.nii.gz"),
    )

    # Coronal LR views for each transform variant
    for suffix in ["_translated", "_rotated", "_translated2", "_rigid"]:
        variant_fname = base.replace(".nii.gz", f"{suffix}.nii.gz")
        output_fname = variant_fname.replace(".nii.gz", "_coronal_LR.nii.gz")
        resample_and_save(input_file=variant_fname, output_file=output_fname, view="coronal")

    # Mean baseline from axial + coronal LR resampled
    axial_fname = base.replace(".nii.gz", "_axial_LR.nii.gz")
    coronal_fname = base.replace(".nii.gz", "_coronal_LR.nii.gz")
    axial_img = resample_nib(nib.load(axial_fname), voxel_spacing=(1, 1, 1))
    coronal_img = resample_nib(nib.load(coronal_fname), voxel_spacing=(1, 1, 1))

    nib.save(axial_img, axial_fname.replace("_LR.nii.gz", "_LR_resampled.nii.gz"))
    nib.save(coronal_img, coronal_fname.replace("_LR.nii.gz", "_LR_resampled.nii.gz"))

    mean_img = nib.Nifti1Image(
        (axial_img.dataobj + coronal_img.dataobj) / 2, axial_img.affine, axial_img.header
    )
    nib.save(mean_img, base.replace(".nii.gz", "_axial_LR_coronal_LR_mean.nii.gz"))


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess OASIS-3 brain MRI scans for SIMS-MRI training."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("OASIS-3-BIDS-selected"),
        help="Root directory of the OASIS-3 BIDS dataset (default: OASIS-3-BIDS-selected)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("oasis_random"),
        help="Output directory for preprocessed subjects (default: oasis_random)",
    )
    args = parser.parse_args()

    participants_file = args.dataset_root / "selected_participants_brain.tsv"
    participants_df = pd.read_csv(participants_file, sep="\t")
    required_columns = {"participant_id", "selected_scan"}
    missing_columns = required_columns.difference(participants_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in TSV: {missing_columns}")

    records = []
    for row in participants_df.itertuples(index=False):
        subject_id = row.participant_id
        rel_scan_path = Path(row.selected_scan)
        full_scan_path = args.dataset_root / rel_scan_path
        if not full_scan_path.exists():
            print(f"[!] Skipping {subject_id}: selected scan not found -> {rel_scan_path}")
            continue
        records.append({"participant_id": subject_id, "selected_scan": str(rel_scan_path)})

    if not records:
        raise RuntimeError("No valid selected scans found. Check the TSV entries.")

    oasis_demo = pd.DataFrame(records)
    image_dir = str(args.dataset_root)

    for row in oasis_demo.itertuples(index=False):
        input_path = str(Path(image_dir) / row.selected_scan)
        try:
            preprocess_oasis_subject(input_path, subject_id=row.participant_id, output_root=args.output_root)
        except Exception as exc:
            print(f"[!] Failed to process {row.participant_id}: {exc}")


if __name__ == "__main__":
    main()
