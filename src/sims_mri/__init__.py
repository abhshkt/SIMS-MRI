"""SIMS-MRI: Single-Subject Multi-View MRI Super-Resolution."""
import pathlib, sys

def _setup_upstream_paths():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if all((repo_root / name).is_dir() for name in ("IDIR", "multi_contrast_inr")):
        root_str = str(repo_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        # Add multi_contrast_inr/ so its bare sibling imports resolve
        mci_str = str(repo_root / "multi_contrast_inr")
        if mci_str not in sys.path:
            sys.path.insert(0, mci_str)

_setup_upstream_paths()
del _setup_upstream_paths
