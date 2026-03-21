"""Phase 2 registration training entrypoint with FreeNeRF-style hash unlock."""

import argparse
import json
import os
import pathlib
import time
from datetime import datetime

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from IDIR.networks import networks
from IDIR.objectives import regularizers
from sims_mri.experiment_utils import (
    default_config_path,
    extract_parent_id_from_path,
    format_number_short,
    generate_unique_id,
    get_image_frame,
    hash_ckpt_for,
    mlflow_log_metrics,
    resolve_mlflow_tracking_uri,
    safe_append_to_csv,
    shorten_project_name,
    snapshot_state_to_cpu,
)
from sims_mri.hash_encoder_wrapper_freenerf import HashEncoderWrapper
from sims_mri.loader import MultiViewDataset
from sims_mri.utils import EarlyStopping
from multi_contrast_inr.loss_functions import NCC, NMI, MILossGaussian
from multi_contrast_inr.model import MLPv1
from multi_contrast_inr.utils import dict2obj

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None



def build_loss_tag(config):
    """Build a compact tag for the configured base loss."""
    base_loss = (
        str(getattr(config.TRAINING, "LOSS", "loss")).lower().replace("loss", "")
    )
    extras = []
    if getattr(config.TRAINING, "USE_MI", False):
        extras.append("mi")
    if getattr(config.TRAINING, "USE_NMI", False):
        extras.append("nmi")
    if getattr(config.TRAINING, "USE_CC", False):
        extras.append("cc")
    parts = [base_loss] + extras
    return "_".join([p for p in parts if p])

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2 Registration-Only Training Script with FreeNeRF-style Progressive Hash Unlock. "
                    "Loads a pre-trained Phase 1 model and runs only Phase 2 registration."
    )

    # Config and basic args
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help="config file (.yaml) containing the hyper-parameters for training.",
    )
    parser.add_argument("--logging", action="store_true")
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: local file store at ./mlruns).",
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default=None,
        help="MLflow experiment name (default: project name).",
    )
    parser.add_argument(
        "--mlflow_run_name",
        type=str,
        default=None,
        help="MLflow run name (default: generated unique experiment ID).",
    )
    parser.add_argument("--batch_size", type=int, default=None)

    # Phase 1 handling
    parser.add_argument(
        "--phase1_checkpoint",
        type=str,
        default=None,
        help="Path to Phase 1 model checkpoint (*_model.pt file). Required unless --run_phase1 is set.",
    )
    parser.add_argument(
        "--run_phase1",
        action="store_true",
        help="Run Phase 1 training first if no checkpoint provided.",
    )
    parser.add_argument(
        "--initial_generation_epochs",
        type=int,
        default=None,
        help="Phase 1 epochs (required if --run_phase1).",
    )

    # Phase 2 specific args
    parser.add_argument(
        "--reg_lr",
        type=float,
        default=1e-5,
        help="Registration learning rate (default: 1e-5).",
    )
    parser.add_argument(
        "--registration_epochs",
        type=int,
        default=None,
        help="Number of registration epochs (default: from config).",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["cc", "mi", "nmi", "mse", "l1"],
        default=None,
        help="Loss function for registration: cc, mi, nmi, mse, l1 (default: inferred from config).",
    )
    parser.add_argument(
        "--regularization",
        type=str,
        choices=["jacobian", "hyper", "bending", "none"],
        default=None,
        help="Regularization type: jacobian, hyper, bending, none (default: from config or none).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Regularization weight (default: from config or 0.1).",
    )
    parser.add_argument(
        "--clamp_coords",
        action="store_true",
        help="Clamp displaced coordinates to [-1, 1] range.",
    )
    parser.add_argument(
        "--disable_early_stopping",
        action="store_true",
        help="Disable early stopping for Phase 2 registration.",
    )

    parser.add_argument(
        "--use_registration_scheduler",
        action="store_true",
        help="Enable LR scheduler for registration (disabled by default to match baseline).",
    )

    # Hash grid encoder arguments
    parser.add_argument(
        "--hash_n_levels",
        type=int,
        default=None,
        help="Number of resolution levels in hash grid (default: from config, fallback 16).",
    )
    parser.add_argument(
        "--hash_n_features_per_level",
        type=int,
        default=None,
        help="Number of features per level (default: from config, fallback 2).",
    )
    parser.add_argument(
        "--hash_log2_size",
        type=int,
        default=None,
        help="Log2 of hash table size (default: from config, fallback 19).",
    )
    parser.add_argument(
        "--hash_base_resolution",
        type=int,
        default=None,
        help="Base resolution at coarsest level (default: from config, fallback 16).",
    )
    parser.add_argument(
        "--hash_per_level_scale",
        type=float,
        default=None,
        help="Resolution multiplier between levels (default: from config, fallback 1.39).",
    )
    parser.add_argument(
        "--concat_coords",
        action="store_true",
        help="Concatenate original coordinates with hash embeddings (recommended).",
    )

    # Progressive hash unlock arguments (FreeNeRF-style)
    parser.add_argument(
        "--progressive_hash_unlock",
        action="store_true",
        help="Enable FreeNeRF-style progressive hash level unlock (coarse-to-fine)",
    )
    parser.add_argument(
        "--hash_unlock_end_fraction",
        type=float,
        default=None,
        help="Fraction of training at which all hash levels are unlocked (default: from config, fallback 0.9)",
    )

    # Dataset specific
    parser.add_argument("--subject_id", type=str, default=None)
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Override project / MLflow experiment name.",
    )
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate for Phase 1 (if running).")

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for Phase 2 artifacts. If not specified, creates new runs/{unique_id}/ directory.",
    )

    return parser.parse_args()


def main(args):
    print(args)

    # Validate arguments
    if not args.run_phase1 and args.phase1_checkpoint is None:
        raise ValueError(
            "Either --phase1_checkpoint must be provided, or --run_phase1 must be set."
        )

    if args.run_phase1 and args.initial_generation_epochs is None:
        raise ValueError(
            "--initial_generation_epochs is required when --run_phase1 is set."
        )

    # Initialize
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Load config
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config_dict)

    # Apply config defaults for hash grid args (CLI overrides config)
    gen_cfg = config.GENERATION_MODEL
    if args.hash_n_levels is None:
        args.hash_n_levels = getattr(gen_cfg, "HASH_N_LEVELS", 16)
    if args.hash_n_features_per_level is None:
        args.hash_n_features_per_level = getattr(gen_cfg, "HASH_N_FEATURES_PER_LEVEL", 2)
    if args.hash_log2_size is None:
        args.hash_log2_size = getattr(gen_cfg, "HASH_LOG2_SIZE", 19)
    if args.hash_base_resolution is None:
        args.hash_base_resolution = getattr(gen_cfg, "HASH_BASE_RESOLUTION", 16)
    if args.hash_per_level_scale is None:
        args.hash_per_level_scale = getattr(gen_cfg, "HASH_PER_LEVEL_SCALE", 1.39)
    if args.hash_unlock_end_fraction is None:
        args.hash_unlock_end_fraction = getattr(gen_cfg, "HASH_UNLOCK_END_FRACTION", 0.9)
    if not args.concat_coords:
        args.concat_coords = getattr(gen_cfg, "CONCAT_COORDS", False)
    if not args.progressive_hash_unlock:
        args.progressive_hash_unlock = getattr(gen_cfg, "PROGRESSIVE_HASH_UNLOCK", False)

    if args.disable_early_stopping:
        if hasattr(config.TRAINING, "EARLY_STOPPING"):
            config.TRAINING.EARLY_STOPPING.ENABLED = False
        config_dict.setdefault("TRAINING", {}).setdefault("EARLY_STOPPING", {})["ENABLED"] = False
        print("Early stopping disabled for Phase 2.")

    # Override config with CLI args
    if args.lr is not None:
        config.TRAINING.LR = args.lr
        config_dict["TRAINING"]["LR"] = args.lr

    if args.batch_size is not None:
        config.TRAINING.BATCH_SIZE = args.batch_size
        config_dict["TRAINING"]["BATCH_SIZE"] = args.batch_size

    if args.initial_generation_epochs is not None:
        config.TRAINING.initial_generation_epochs = args.initial_generation_epochs
        config_dict["TRAINING"]["initial_generation_epochs"] = args.initial_generation_epochs

    if args.registration_epochs is not None:
        config.TRAINING.registration_epochs = args.registration_epochs
        config_dict["TRAINING"]["registration_epochs"] = args.registration_epochs

    if args.subject_id is not None:
        config.DATASET.SUBJECT_ID = args.subject_id
        config_dict["DATASET"]["SUBJECT_ID"] = args.subject_id

    if args.project is not None:
        config.SETTINGS.PROJECT_NAME = args.project
        config_dict["SETTINGS"]["PROJECT_NAME"] = args.project

    # Set up loss configuration based on --loss argument (infer from config if not provided)
    if args.loss is None:
        # Infer loss from config
        if getattr(config.TRAINING, "USE_CC", False):
            args.loss = "cc"
        elif getattr(config.TRAINING, "USE_MI", False):
            args.loss = "mi"
        elif getattr(config.TRAINING, "USE_NMI", False):
            args.loss = "nmi"
        elif config.TRAINING.LOSS == "L1Loss":
            args.loss = "l1"
        else:
            args.loss = "mse"  # default fallback
        print(f"Loss inferred from config: {args.loss}")

    config.TRAINING.USE_CC = args.loss == "cc"
    config.TRAINING.USE_MI = args.loss == "mi"
    config.TRAINING.USE_NMI = args.loss == "nmi"
    config_dict["TRAINING"]["USE_CC"] = config.TRAINING.USE_CC
    config_dict["TRAINING"]["USE_MI"] = config.TRAINING.USE_MI
    config_dict["TRAINING"]["USE_NMI"] = config.TRAINING.USE_NMI

    # Resolve regularization and alpha: CLI overrides config, config overrides defaults
    if args.regularization is None:
        args.regularization = getattr(
            getattr(config, "REGISTRATION_MODEL", None), "REGULARIZATION", "none"
        ) or "none"
    if args.alpha is None:
        args.alpha = getattr(
            getattr(config, "REGISTRATION_MODEL", None), "ALPHA", 0.1
        ) or 0.1

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Generate unique experiment ID
    unique_id = generate_unique_id()
    print(f"Experiment ID: {unique_id}")

    # Extract parent Phase 1 ID
    parent_phase1_id = None
    if args.phase1_checkpoint:
        parent_phase1_id = extract_parent_id_from_path(args.phase1_checkpoint)
        print(f"Parent Phase 1 ID: {parent_phase1_id}")

    # MLflow setup
    mlflow_run_id = None
    mlflow_tracking_uri = None
    mlflow_experiment_name = args.mlflow_experiment or config.SETTINGS.PROJECT_NAME
    if args.logging:
        if mlflow is None:
            print("mlflow is not installed -> skipping MLflow logging")
            args.logging = False
        else:
            print("MLflow logging enabled")
            try:
                mlflow_tracking_uri = resolve_mlflow_tracking_uri(args.mlflow_tracking_uri)
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                mlflow.set_experiment(mlflow_experiment_name)

                run_name = args.mlflow_run_name or unique_id
                mlflow.start_run(
                    run_name=run_name,
                    tags={
                        "unique_experiment_id": unique_id,
                        "parent_phase1_id": str(parent_phase1_id) if parent_phase1_id else "none",
                        "subject_id": str(config.DATASET.SUBJECT_ID),
                        "script_name": "phase2_registration.py",
                        "phase2_only": "true",
                        "progressive_hash_unlock": str(args.progressive_hash_unlock),
                    },
                )

                active_run = mlflow.active_run()
                mlflow_run_id = active_run.info.run_id if active_run else None

                # Log parameters
                mlflow_params = {
                    "unique_id": unique_id,
                    "parent_phase1_id": parent_phase1_id if parent_phase1_id else "none",
                    "subject_id": config.DATASET.SUBJECT_ID,
                    "batch_size": config.TRAINING.BATCH_SIZE,
                    "registration_epochs": config.TRAINING.registration_epochs,
                    "reg_lr": args.reg_lr,
                    "loss": args.loss,
                    "regularization": args.regularization,
                    "alpha": args.alpha,
                    "clamp_coords": args.clamp_coords,
                    "hash_n_levels": args.hash_n_levels,
                    "hash_n_features_per_level": args.hash_n_features_per_level,
                    "hash_log2_size": args.hash_log2_size,
                    "hash_base_resolution": args.hash_base_resolution,
                    "hash_per_level_scale": args.hash_per_level_scale,
                    "concat_coords": args.concat_coords,
                    "progressive_hash_unlock": args.progressive_hash_unlock,
                    "hash_unlock_end_fraction": args.hash_unlock_end_fraction,
                    "disable_early_stopping": args.disable_early_stopping,
                    "run_phase1": args.run_phase1,
                    "phase1_checkpoint": args.phase1_checkpoint if args.phase1_checkpoint else "N/A",
                    "seed": config.TRAINING.SEED,
                }
                mlflow.log_params({k: str(v) for k, v in mlflow_params.items()})
            except Exception as e:
                print(f"MLflow setup failed ({e}) -> skipping MLflow logging")
                try:
                    if mlflow.active_run() is not None:
                        mlflow.end_run()
                except Exception:
                    pass
                mlflow_run_id = None
                mlflow_tracking_uri = None
                args.logging = False

    # Create output directories
    # Determine base directory: use --output_dir if provided, otherwise create new unique dir
    if args.output_dir:
        exp_base_dir = args.output_dir
        # Ensure the output directory exists
        pathlib.Path(exp_base_dir).mkdir(parents=True, exist_ok=True)
        print(f"Using output directory: {exp_base_dir} (Phase 1 co-location mode)")
    else:
        exp_base_dir = f"runs/{unique_id}"
        print(f"Creating new experiment directory: {exp_base_dir}")

    proj_short = shorten_project_name(config.SETTINGS.PROJECT_NAME)

    hash_tag = f"hash_L{args.hash_n_levels}_F{args.hash_n_features_per_level}"
    if args.concat_coords:
        hash_tag += "_cat"

    es_suffix = (
        "_es"
        if hasattr(config.TRAINING, "EARLY_STOPPING")
        and config.TRAINING.EARLY_STOPPING.ENABLED
        else ""
    )

    # Add _phase2_{unique_id} suffix to subdirectories when using output_dir
    # This avoids collisions with Phase 1 AND supports multiple Phase 2 runs
    if args.output_dir:
        phase2_suffix = f"_phase2_{unique_id}"
    else:
        phase2_suffix = ""
    weight_subdir = f"{hash_tag}_{proj_short}_w{es_suffix}{phase2_suffix}"
    image_subdir = f"{hash_tag}_{proj_short}_img{es_suffix}{phase2_suffix}"

    weight_dir = os.path.join(exp_base_dir, weight_subdir)
    image_dir = os.path.join(exp_base_dir, image_subdir)

    pathlib.Path(weight_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)

    # Log experiment details to central CSV
    csv_fieldnames = [
        "unique_id",
        "timestamp",
        "script_name",
        "parent_phase1_id",
        "subject_id",
        "lr_contrast1",
        "lr_contrast2",
        "batch_size",
        "registration_epochs",
        "reg_lr",
        "loss",
        "regularization",
        "alpha",
        "clamp_coords",
        "hash_n_levels",
        "hash_n_features_per_level",
        "hash_log2_size",
        "hash_base_resolution",
        "hash_per_level_scale",
        "concat_coords",
        "progressive_hash_unlock",
        "hash_unlock_end_fraction",
        "mlflow_run_id",
        "mlflow_tracking_uri",
        "mlflow_experiment",
        "weight_dir",
        "image_dir",
        "config_path",
        "phase1_checkpoint",
        "seed",
        "device",
        "notes",
    ]

    exp_log_data = {
        "unique_id": unique_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script_name": "phase2_registration.py",
        "parent_phase1_id": parent_phase1_id if parent_phase1_id else "N/A",
        "subject_id": config.DATASET.SUBJECT_ID,
        "lr_contrast1": config.DATASET.LR_CONTRAST1,
        "lr_contrast2": config.DATASET.LR_CONTRAST2,
        "batch_size": config.TRAINING.BATCH_SIZE,
        "registration_epochs": config.TRAINING.registration_epochs,
        "reg_lr": args.reg_lr,
        "loss": args.loss,
        "regularization": args.regularization,
        "alpha": args.alpha,
        "clamp_coords": args.clamp_coords,
        "hash_n_levels": args.hash_n_levels,
        "hash_n_features_per_level": args.hash_n_features_per_level,
        "hash_log2_size": args.hash_log2_size,
        "hash_base_resolution": args.hash_base_resolution,
        "hash_per_level_scale": args.hash_per_level_scale,
        "concat_coords": args.concat_coords,
        "progressive_hash_unlock": args.progressive_hash_unlock,
        "hash_unlock_end_fraction": args.hash_unlock_end_fraction,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "mlflow_experiment": mlflow_experiment_name,
        "weight_dir": weight_dir,
        "image_dir": image_dir,
        "config_path": args.config,
        "phase1_checkpoint": args.phase1_checkpoint if args.phase1_checkpoint else "N/A",
        "seed": config.TRAINING.SEED,
        "device": str(device),
        "notes": f"phase2_only_loss-{args.loss}_reg-{args.regularization}_alpha-{args.alpha}_freenerf",
    }

    pathlib.Path("runs").mkdir(exist_ok=True)
    csv_path = "runs/experiment_log_phase2_freenerf.csv"
    if not safe_append_to_csv(csv_path, exp_log_data, csv_fieldnames):
        # Use unique naming for metadata fallback when using output_dir
        if args.output_dir:
            metadata_filename = f"metadata_phase2_{unique_id}.json"
        else:
            metadata_filename = "metadata.json"
        print(f"Warning: Could not write to central CSV, saving metadata to {exp_base_dir}/{metadata_filename}")
        with open(os.path.join(exp_base_dir, metadata_filename), "w") as f:
            json.dump(exp_log_data, f, indent=2)

    # Save full config snapshot
    config_snapshot = {
        "unique_id": unique_id,
        "parent_phase1_id": parent_phase1_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "config": config_dict,
        "mlflow_info": {
            "run_id": mlflow_run_id,
            "tracking_uri": mlflow_tracking_uri,
            "experiment": mlflow_experiment_name,
        },
    }
    # Save config snapshot with unique naming to avoid overwriting
    # When using output_dir: include unique_id to support multiple Phase 2 runs
    if args.output_dir:
        config_filename = f"config_snapshot_phase2_{unique_id}.json"
    else:
        config_filename = "config_snapshot.json"
    with open(os.path.join(exp_base_dir, config_filename), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # Log config artifacts to MLflow
    if args.logging and mlflow is not None and mlflow.active_run() is not None:
        mlflow.set_tags({
            "weight_dir": weight_dir,
            "image_dir": image_dir,
            "config_path": args.config,
        })
        mlflow.log_dict(config_snapshot, "config_snapshot.json")
        mlflow.log_dict(config_dict, "config_dict.json")
        try:
            mlflow.log_artifact(args.config, artifact_path="config")
        except (OSError, ValueError):
            pass

    # Seeding
    torch.manual_seed(config.TRAINING.SEED)
    np.random.seed(config.TRAINING.SEED)

    # Load dataset
    dataset = MultiViewDataset(
        image_dir=config.SETTINGS.DIRECTORY,
        name=config.SETTINGS.PROJECT_NAME,
        subject_id=config.DATASET.SUBJECT_ID,
        contrast1_LR_str=config.DATASET.LR_CONTRAST1,
        contrast2_LR_str=config.DATASET.LR_CONTRAST2,
    )

    output_size = 1

    # Initialize hash grid encoder with progressive unlock support
    input_mapper = HashEncoderWrapper(
        n_levels=args.hash_n_levels,
        n_features_per_level=args.hash_n_features_per_level,
        log2_hashmap_size=args.hash_log2_size,
        base_resolution=args.hash_base_resolution,
        per_level_scale=args.hash_per_level_scale,
        concat_coords=args.concat_coords,
        input_range=(-1, 1),
        progressive_unlock=args.progressive_hash_unlock,
        unlock_end_fraction=args.hash_unlock_end_fraction,
    ).to(device)
    input_size = input_mapper.output_dim

    print(f"Hash grid encoder initialized:")
    print(f"  - Levels: {args.hash_n_levels}")
    print(f"  - Features per level: {args.hash_n_features_per_level}")
    print(f"  - Concat coords: {args.concat_coords}")
    print(f"  - Output dimension: {input_size}")
    print(f"  - Progressive unlock: {args.progressive_hash_unlock}")
    if args.progressive_hash_unlock:
        print(f"  - Unlock end fraction: {args.hash_unlock_end_fraction}")

    # Initialize image generation model
    if config.GENERATION_MODEL.TYPE == "mlp":
        model = MLPv1(
            input_size=input_size,
            output_size=output_size,
            hidden_size=1024,
            num_layers=4,
            dropout=0,
        )
    elif config.GENERATION_MODEL.TYPE == "siren":
        model = networks.Siren([3, 1024, 1024, 1024, 1024, 1], True, 15)
    else:
        raise ValueError(f"Unknown generation model type: {config.GENERATION_MODEL.TYPE}")

    # Basic loss criterion for Phase 1 (if needed)
    if config.TRAINING.LOSS == "L1Loss":
        criterion = nn.L1Loss()
    elif config.TRAINING.LOSS == "MSELoss":
        criterion = nn.MSELoss()
    else:
        criterion = nn.MSELoss()  # default

    # Registration loss setup based on --loss argument
    if args.loss == "cc":
        reg_criterion = NCC()
    elif args.loss == "mi":
        reg_criterion = MILossGaussian(num_bins=32, sample_ratio=1, gt_val=0.471)
    elif args.loss == "nmi":
        reg_criterion = NMI(intensity_range=(0, 1), nbins=32, sigma=0.1)
    elif args.loss == "mse":
        reg_criterion = nn.MSELoss()
    elif args.loss == "l1":
        reg_criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type: {args.loss}")

    # Initialize registration model
    if hasattr(config, "REGISTRATION_MODEL"):
        if config.REGISTRATION_MODEL.TYPE == "mlp":
            rotation_param = nn.Linear(512, 512)
            rotation_param = rotation_param.to(device)
        elif config.REGISTRATION_MODEL.TYPE == "siren":
            rotation_param = networks.Siren([3, 256, 256, 256, 3], True, 32)
            rotation_param = rotation_param.to(device)
        else:
            raise ValueError(f"Unknown registration model type: {config.REGISTRATION_MODEL.TYPE}")
    else:
        raise ValueError("REGISTRATION_MODEL must be defined in config for Phase 2")

    # Build compact model name
    gen_tag = "mlp"
    if config.GENERATION_MODEL.TYPE == "siren":
        gen_tag = "siren"
    elif config.GENERATION_MODEL.TYPE not in ("mlp",):
        gen_tag = str(config.GENERATION_MODEL.TYPE)

    reg_tag = "reg"
    if config.REGISTRATION_MODEL.TYPE == "siren":
        reg_tag = "reg-siren"
    elif config.REGISTRATION_MODEL.TYPE == "mlp":
        reg_tag = "reg-mlp"
    else:
        reg_tag = f"reg-{config.REGISTRATION_MODEL.TYPE}"

    # Add regularization tag
    reg_reg_short = {"jacobian": "jac", "hyper": "hyp", "bending": "bend", "none": "noreg"}.get(
        args.regularization, args.regularization
    )
    reg_tag = f"{reg_tag}-{reg_reg_short}"

    loss_tag = args.loss
    model_name = (
        f"{config.DATASET.SUBJECT_ID}_"
        f"proj-{proj_short}_"
        f"{hash_tag}_"
        f"{gen_tag}_"
        f"{reg_tag}_"
        f"{loss_tag}_"
        f"b{config.TRAINING.BATCH_SIZE}"
    )

    # Print parameter counts
    print(f"Number of MLP parameters {sum(p.numel() for p in model.parameters())}")
    print(f"Number of hash encoder parameters {sum(p.numel() for p in input_mapper.parameters())}")
    print(f"Number of registration parameters {sum(p.numel() for p in rotation_param.parameters())}")

    # Load training data
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.TRAINING.BATCH_SIZE,
        shuffle=config.TRAINING.SHUFFLING,
        num_workers=config.SETTINGS.NUM_WORKERS,
    )

    # Move model to device
    model = model.to(device)

    ################ PHASE 1 (if --run_phase1) #######################
    epoch_rough_image_generation = config.TRAINING.initial_generation_epochs if args.run_phase1 else 0

    if args.run_phase1 and args.phase1_checkpoint is None:
        print("=" * 60)
        print("Running Phase 1 training first...")
        print("=" * 60)

        # Compute total steps for Phase 1 progressive unlock
        total_steps_phase1 = epoch_rough_image_generation * len(train_dataloader)

        # Initialize Phase 1 optimizer
        params_img = list(model.parameters()) + list(input_mapper.parameters())
        optimizer_img = torch.optim.AdamW(params_img, lr=config.TRAINING.LR * 3, weight_decay=5e-5)
        # Match baseline T_max: initial_generation_epochs + generation_epochs
        phase1_t_max = epoch_rough_image_generation + getattr(config.TRAINING, "generation_epochs", 0)
        scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer_img, T_max=phase1_t_max
        )

        # Initialize early stopping for image generation phase
        early_stopping_enabled = (
            hasattr(config.TRAINING, "EARLY_STOPPING")
            and config.TRAINING.EARLY_STOPPING.ENABLED
        )
        if early_stopping_enabled:
            # Get delta_fraction from config (used for improvement threshold)
            es_delta_fraction = getattr(
                config.TRAINING.EARLY_STOPPING, "DELTA_FRACTION",
                config.TRAINING.EARLY_STOPPING.DELTA
            )
            early_stop_img = EarlyStopping(
                patience=config.TRAINING.EARLY_STOPPING.PATIENCE,
                delta_fraction=es_delta_fraction,
                mode="min",
                verbose=config.TRAINING.EARLY_STOPPING.VERBOSE,
            )
            # Track best weights in memory (save only once at early stop)
            best_img_model_state = None
            best_img_hash_state = None
            best_img_epoch = None

        # Track actual saved checkpoint epoch (may differ from final if early stopped)
        saved_phase1_epoch = epoch_rough_image_generation - 1  # default to final epoch

        model.train()
        for epoch in range(epoch_rough_image_generation):
            epoch_metrics = {}

            model_name_epoch = f"{model_name}_e{int(epoch)}_model.pt"
            model_path = os.path.join(weight_dir, model_name_epoch)
            print(f"Training image generation model: epoch {epoch}")

            loss_epoch = 0.0
            start = time.time()

            for batch_idx, (d1, d2) in enumerate(train_dataloader):
                # Set training progress for progressive hash unlock
                current_step = epoch * len(train_dataloader) + batch_idx
                input_mapper.set_training_progress(current_step / total_steps_phase1)

                batch_metrics = {}
                loss_batch = 0

                (data1, label1) = d1
                (data2, label2) = d2

                contrast1_mask = label1[:, 0] != -1.0
                contrast1_labels = label1[contrast1_mask, 0]
                contrast1_labels = contrast1_labels.reshape(-1, 1).to(device=device)
                contrast1_data = data1[contrast1_mask, :].to(device=device)

                # Apply hash grid encoding
                contrast1_data = input_mapper(contrast1_data)

                optimizer_img.zero_grad()

                target = model(contrast1_data)

                mse_target1 = target[: len(contrast1_data)]
                loss_mse = criterion(mse_target1, contrast1_labels)

                phase1_multiplier = (
                    config.TRAINING.PHASE1_LOSS_MULTIPLIER
                    if hasattr(config.TRAINING, "PHASE1_LOSS_MULTIPLIER")
                    else 1.0
                )
                loss_mse = loss_mse * phase1_multiplier

                if args.logging:
                    img_step = epoch * len(train_dataloader) + batch_idx
                    batch_metrics.update({"img_loss": loss_mse.item()})
                    batch_metrics.update({"phase1_loss_multiplier": phase1_multiplier})

                    # Log progressive hash unlock metrics
                    if args.progressive_hash_unlock:
                        level_weights = input_mapper.compute_level_weights()
                        active_levels = (level_weights > 0.5).sum().item()
                        batch_metrics.update({
                            "hash_unlock_progress": current_step / total_steps_phase1,
                            "hash_active_levels": active_levels,
                        })

                    mlflow_log_metrics(mlflow, batch_metrics, step=img_step)

                loss_mse.backward()
                optimizer_img.step()

                loss_batch = loss_mse.item()
                loss_epoch += loss_batch

                if args.logging:
                    batch_metrics.update({"img_batch_loss": loss_batch})
                    mlflow_log_metrics(mlflow, batch_metrics, step=img_step)

            epoch_time = time.time() - start
            lr = optimizer_img.param_groups[0]["lr"]
            avg_loss = loss_epoch / len(train_dataloader)

            if args.logging:
                epoch_metrics.update({"img_epoch_no": epoch})
                epoch_metrics.update({"img_epoch_time": epoch_time})
                epoch_metrics.update({"img_epoch_loss": loss_epoch})
                epoch_metrics.update({"img_avg_loss": avg_loss})
                epoch_metrics.update({"img_lr": lr})
                mlflow_log_metrics(mlflow, epoch_metrics, step=epoch)

            scheduler_img.step()

            # Check early stopping (no intermediate saves - only final checkpoint)
            if early_stopping_enabled:
                # Call early stopping first to update its state
                should_stop = early_stop_img(avg_loss)

                # Track best weights in memory when loss improves (use EarlyStopping.improved)
                if early_stop_img.improved:
                    best_img_epoch = epoch
                    best_img_model_state = snapshot_state_to_cpu(model)
                    best_img_hash_state = snapshot_state_to_cpu(input_mapper)

                if should_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    print(f"  Best loss was at epoch {best_img_epoch} (loss: {early_stop_img.best_loss:.6f})")
                    if args.logging:
                        mlflow_log_metrics(mlflow, {"img_early_stop_epoch": epoch}, step=epoch)
                        mlflow_log_metrics(mlflow, {"img_best_epoch": best_img_epoch}, step=epoch)
                    # Save the BEST weights once
                    saved_phase1_epoch = best_img_epoch
                    best_model_path = os.path.join(weight_dir, f"{model_name}_e{best_img_epoch}_model.pt")
                    torch.save(best_img_model_state, best_model_path)
                    torch.save(best_img_hash_state, hash_ckpt_for(best_model_path))
                    # Load best weights for inference
                    model.load_state_dict(best_img_model_state)
                    input_mapper.load_state_dict(best_img_hash_state)
                    break

            # Save checkpoint at final epoch (only if we didn't early stop)
            if epoch == (epoch_rough_image_generation - 1):
                torch.save(model.state_dict(), model_path)
                torch.save(input_mapper.state_dict(), hash_ckpt_for(model_path))

        # Use actual saved checkpoint epoch for naming
        rough_generation_model_name_epoch = f"{model_name}_e{int(saved_phase1_epoch)}_model.pt"
        phase1_model_path = os.path.join(weight_dir, rough_generation_model_name_epoch)
        print(f"Phase 1 training complete. Model saved to: {phase1_model_path}")

        ################ INFERENCE for Phase 1 #######################
        infer_batch_size = 10000

        (out_affine1, out_loader1, out_image1, out_dim_xyz1) = get_image_frame(
            dataset, config, batch_size=infer_batch_size, input_idx=1
        )

        model.eval()

        init_generation_fname = os.path.join(
            image_dir,
            rough_generation_model_name_epoch.replace("model.pt", f"_ct1_init.nii.gz"),
        )

        if not os.path.exists(init_generation_fname):
            for batch_idx, (data) in enumerate(out_loader1):
                data = data.to(device)
                data = input_mapper(data)

                output = model(data)

                out_image1[
                    batch_idx * infer_batch_size : (batch_idx * infer_batch_size + len(output)),
                    :,
                ] = output.cpu().detach().numpy()

            print("Generating NIFTIs for contrast1")
            (x_dim, y_dim, z_dim) = out_dim_xyz1
            scaler = MinMaxScaler()
            label_arr = np.array(out_image1, dtype=np.float32)
            model_intensities_contrast1 = scaler.fit_transform(label_arr.reshape(-1, 1))
            img_contrast1 = model_intensities_contrast1.reshape((x_dim, y_dim, z_dim))
            img = nib.Nifti1Image(img_contrast1, out_affine1)
            nib.save(img, init_generation_fname)

    else:
        # Load Phase 1 checkpoint
        print("=" * 60)
        print(f"Loading Phase 1 checkpoint: {args.phase1_checkpoint}")
        print("=" * 60)

        model.load_state_dict(torch.load(args.phase1_checkpoint, map_location=device, weights_only=False))
        hash_ckpt = hash_ckpt_for(args.phase1_checkpoint)
        if os.path.exists(hash_ckpt):
            input_mapper.load_state_dict(torch.load(hash_ckpt, map_location=device, weights_only=False))
            print(f"Loaded hash encoder from: {hash_ckpt}")
        else:
            print(f"Warning: Hash encoder checkpoint not found at {hash_ckpt}; using initialized weights.")

    ################ PHASE 2: REGISTRATION TRAINING #######################
    print("=" * 60)
    print("Starting Phase 2 Registration Training...")
    print(f"  Loss: {args.loss}")
    print(f"  Regularization: {args.regularization} (alpha={args.alpha})")
    print(f"  Learning rate: {args.reg_lr}")
    print(f"  Epochs: {config.TRAINING.registration_epochs}")
    print("=" * 60)

    # Setup registration optimizer
    params_reg = list(rotation_param.parameters())
    optimizer_reg = torch.optim.Adam(params_reg, lr=args.reg_lr)

    # Setup scheduler
    if args.use_registration_scheduler:
        scheduler_reg = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer_reg, T_max=config.TRAINING.registration_epochs
        )

    # Initialize early stopping for registration phase
    early_stopping_enabled = (
        hasattr(config.TRAINING, "EARLY_STOPPING")
        and config.TRAINING.EARLY_STOPPING.ENABLED
    )
    if early_stopping_enabled:
        # Get delta_fraction from config (used for improvement threshold)
        es_delta_fraction_reg = getattr(
            config.TRAINING.EARLY_STOPPING, "DELTA_FRACTION",
            config.TRAINING.EARLY_STOPPING.DELTA
        )
        early_stop_reg = EarlyStopping(
            patience=config.TRAINING.EARLY_STOPPING.PATIENCE_REGISTRATION,
            delta_fraction=es_delta_fraction_reg,
            mode="min",
            verbose=config.TRAINING.EARLY_STOPPING.VERBOSE,
        )
        # Track best weights in memory (save only once at early stop)
        best_reg_epoch = None
        best_rotation_state = None

    # Training mode
    model.eval()
    input_mapper.eval()  # Disable progressive hash mask so all levels are active

    rotation_param.train()

    epoch_start = epoch_rough_image_generation if args.run_phase1 else 0
    epoch_end = epoch_start + config.TRAINING.registration_epochs

    # Track actual saved checkpoint epoch (may differ from final if early stopped)
    saved_registration_epoch = epoch_end - 1  # default to final epoch

    for epoch in range(epoch_start, epoch_end):
        epoch_metrics = {}

        model_name_epoch = f"{model_name}_e{int(epoch)}_model.pt"
        model_path = os.path.join(weight_dir, model_name_epoch)

        loss_epoch = 0.0
        start = time.time()

        for batch_idx, (d1, d2) in enumerate(train_dataloader):
            batch_metrics = {}
            loss_batch = 0

            (data1, label1) = d1
            (data2, label2) = d2

            contrast2_mask = label2[:, 0] != -1.0
            contrast2_labels = label2[contrast2_mask, 0]
            contrast2_labels = contrast2_labels.reshape(-1, 1).to(device=device)
            contrast2_data = data2[contrast2_mask, :].to(device=device)
            contrast2_data = contrast2_data.requires_grad_(True)

            if config.REGISTRATION_MODEL.TYPE == "mlp":
                data = input_mapper(contrast2_data)
                data = rotation_param(data)
            elif config.REGISTRATION_MODEL.TYPE == "siren":
                displacement = rotation_param(contrast2_data)
                data = torch.add(displacement, contrast2_data)
                if args.clamp_coords:
                    data = torch.clamp(data, min=-1.0, max=1.0)
                data = input_mapper(data)
            else:
                data = torch.mm(contrast2_data, rotation_param)
                data = input_mapper(data)

            target = model(data)
            rigid_target2 = target

            # Compute registration loss based on --loss argument
            if args.loss == "cc":
                loss = reg_criterion(rigid_target2, contrast2_labels)
            elif args.loss in ["mi", "nmi"]:
                loss = 1 - reg_criterion(rigid_target2.T[None, :], contrast2_labels.T[None, :])
            else:  # mse, l1
                loss = reg_criterion(rigid_target2.T[None, :], contrast2_labels.T[None, :])

            # Add regularization for SIREN registration model
            if config.REGISTRATION_MODEL.TYPE == "siren" and args.regularization != "none":
                output_rel = torch.subtract(displacement, contrast2_data)

                if args.regularization == "jacobian":
                    loss += args.alpha * regularizers.compute_jacobian_loss(
                        contrast2_data, output_rel, batch_size=config.TRAINING.BATCH_SIZE
                    )
                elif args.regularization == "hyper":
                    loss += args.alpha * regularizers.compute_hyper_elastic_loss(
                        contrast2_data, output_rel, batch_size=config.TRAINING.BATCH_SIZE
                    )
                elif args.regularization == "bending":
                    loss += args.alpha * regularizers.compute_bending_energy(
                        contrast2_data, output_rel, batch_size=config.TRAINING.BATCH_SIZE
                    )

            optimizer_reg.zero_grad()
            loss.backward()
            optimizer_reg.step()

            loss_batch = loss.item()
            loss_epoch += loss_batch

            if args.logging:
                reg_step = epoch * len(train_dataloader) + batch_idx
                batch_metrics.update({"reg_batch_loss": loss_batch})

                mlflow_log_metrics(mlflow, batch_metrics, step=reg_step)

        epoch_time = time.time() - start
        lr = optimizer_reg.param_groups[0]["lr"]
        avg_loss = loss_epoch / len(train_dataloader)

        if args.logging:
            epoch_metrics.update({"reg_epoch_no": epoch})
            epoch_metrics.update({"reg_epoch_time": epoch_time})
            epoch_metrics.update({"reg_epoch_loss": loss_epoch})
            epoch_metrics.update({"reg_avg_loss": avg_loss})
            epoch_metrics.update({"reg_lr": lr})

            mlflow_log_metrics(mlflow, epoch_metrics, step=epoch)

        if args.use_registration_scheduler:
            scheduler_reg.step()

        # Check early stopping (no intermediate saves - only final checkpoint)
        if early_stopping_enabled:
            # Call early stopping first to update its state
            should_stop = early_stop_reg(avg_loss)

            # Track best weights in memory when loss improves (use EarlyStopping.improved)
            if early_stop_reg.improved:
                best_reg_epoch = epoch
                best_rotation_state = snapshot_state_to_cpu(rotation_param)

            if should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"  Best loss was at epoch {best_reg_epoch} (loss: {early_stop_reg.best_loss:.6f})")
                if args.logging:
                    mlflow_log_metrics(mlflow, {"reg_early_stop_epoch": epoch}, step=epoch)
                    mlflow_log_metrics(mlflow, {"reg_best_epoch": best_reg_epoch}, step=epoch)
                # Save the BEST weights once
                saved_registration_epoch = best_reg_epoch
                best_ckpt_path = os.path.join(weight_dir, f"{model_name}_e{best_reg_epoch}_model.pt")
                torch.save(best_rotation_state, best_ckpt_path.replace(".pt", "_rotation.pt"))
                # Load best weights for inference
                rotation_param.load_state_dict(best_rotation_state)
                break

        # Save checkpoint at final epoch (only if we didn't early stop)
        if epoch == (epoch_end - 1):
            print(f"Saving final registration model: {model_path}")
            torch.save(rotation_param.state_dict(), model_path.replace(".pt", "_rotation.pt"))

    ################ INFERENCE for registration #######################
    print("=" * 60)
    print("Running Phase 2 Inference...")
    print("=" * 60)

    infer_batch_size = 10000

    (out_affine2, out_loader2, out_image2, out_dim_xyz2) = get_image_frame(
        dataset, config, batch_size=infer_batch_size, input_idx=2
    )

    model.eval()
    input_mapper.eval()  # Ensure progressive hash mask is disabled for inference
    rotation_param_eval = rotation_param.eval()

    registration_model_name_epoch = f"{model_name}_e{int(saved_registration_epoch)}_model_rotation.pt"
    init_registration_fname = os.path.join(
        image_dir,
        registration_model_name_epoch.replace("model_rotation.pt", f"_ct2_reg.nii.gz"),
    )

    if not os.path.exists(init_registration_fname):
        for batch_idx, (data) in enumerate(out_loader2):
            data = data.to(device)

            if config.REGISTRATION_MODEL.TYPE == "mlp":
                data = input_mapper(data)
                data = rotation_param_eval(data)
            elif config.REGISTRATION_MODEL.TYPE == "siren":
                displacement = rotation_param_eval(data)
                data = torch.add(displacement, data)
                if args.clamp_coords:
                    data = torch.clamp(data, min=-1.0, max=1.0)
                data = input_mapper(data)
            else:
                data = torch.mm(data, rotation_param_eval)
                data = input_mapper(data)

            output = model(data)
            out_image2[
                batch_idx * infer_batch_size : (batch_idx * infer_batch_size + len(output)),
                :,
            ] = output.cpu().detach().numpy()

        print("Generating NIFTIs for contrast2 (registered)")
        x_dim, y_dim, z_dim = out_dim_xyz2
        scaler = MinMaxScaler()
        label_arr = np.array(out_image2, dtype=np.float32)
        model_intensities_contrast2 = scaler.fit_transform(label_arr.reshape(-1, 1))
        img_contrast2 = model_intensities_contrast2.reshape((x_dim, y_dim, z_dim))
        img = nib.Nifti1Image(img_contrast2, out_affine2)
        nib.save(img, init_registration_fname)
        print(f"Saved registered image to: {init_registration_fname}")

    # Finish MLflow logging session
    if args.logging and mlflow is not None and mlflow.active_run() is not None:
        mlflow.end_run()

    print("=" * 60)
    print("Phase 2 Registration Training Complete!")
    print(f"Experiment ID: {unique_id}")
    print(f"Parent Phase 1 ID: {parent_phase1_id}")
    print(f"Output directory: {exp_base_dir}")
    print(f"Weights directory: {weight_dir}")
    print(f"Images directory: {image_dir}")
    print("=" * 60)


def _cli():
    main(parse_args())


if __name__ == "__main__":
    _cli()
