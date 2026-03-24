"""Phase 1 image-generation training entrypoint with FreeNeRF-style hash unlock."""

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
from sims_mri.experiment_utils import (
    default_config_path,
    generate_unique_id,
    get_image_frame,
    hash_ckpt_for,
    mlflow_log_metrics,
    resolve_mlflow_tracking_uri,
    safe_append_to_csv,
    shorten_project_name,
)
from sims_mri.hash_encoder_wrapper_freenerf import HashEncoderWrapper
from sims_mri.loader import MultiViewDataset
from sims_mri.utils import EarlyStopping
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
    return base_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1 Image Generation Only Training Script with FreeNeRF-style Progressive Hash Unlock. "
                    "Trains only the image generation model without registration."
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

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of Phase 1 generation epochs (overrides config initial_generation_epochs).",
    )
    parser.add_argument(
        "--initial_generation_epochs",
        type=int,
        default=None,
        help="Alias for --epochs (for compatibility with multi_view_inr_hash_grid_mlflow.py).",
    )
    # Hash grid encoder arguments
    parser.add_argument(
        "--hash_n_levels",
        type=int,
        default=None,
        help="Number of resolution levels in hash grid (default: from config, fallback 16)",
    )
    parser.add_argument(
        "--hash_n_features_per_level",
        type=int,
        default=None,
        help="Number of features per level (default: from config, fallback 2)",
    )
    parser.add_argument(
        "--hash_log2_size",
        type=int,
        default=None,
        help="Log2 of hash table size (default: from config, fallback 19)",
    )
    parser.add_argument(
        "--hash_base_resolution",
        type=int,
        default=None,
        help="Base resolution at coarsest level (default: from config, fallback 16)",
    )
    parser.add_argument(
        "--hash_per_level_scale",
        type=float,
        default=None,
        help="Resolution multiplier between levels (default: from config, fallback 1.39)",
    )
    parser.add_argument(
        "--concat_coords",
        action="store_true",
        help="Concatenate original coordinates with hash embeddings (recommended)",
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
    parser.add_argument(
        "--disable_early_stopping",
        action="store_true",
        help="Disable early stopping for Phase 1 generation training.",
    )

    # Dataset specific
    parser.add_argument("--subject_id", type=str, default=None)
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Override project / MLflow experiment name (default: from config file)",
    )

    return parser.parse_args()


def main(args):
    print(args)

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
        print("Early stopping disabled for Phase 1.")

    # Override config with CLI args
    if args.lr is not None:
        config.TRAINING.LR = args.lr
        config_dict["TRAINING"]["LR"] = args.lr

    if args.batch_size is not None:
        config.TRAINING.BATCH_SIZE = args.batch_size
        config_dict["TRAINING"]["BATCH_SIZE"] = args.batch_size

    # Handle epochs argument (--epochs takes priority over --initial_generation_epochs)
    if args.epochs is not None:
        config.TRAINING.initial_generation_epochs = args.epochs
        config_dict["TRAINING"]["initial_generation_epochs"] = args.epochs
    elif args.initial_generation_epochs is not None:
        config.TRAINING.initial_generation_epochs = args.initial_generation_epochs
        config_dict["TRAINING"]["initial_generation_epochs"] = args.initial_generation_epochs

    if args.subject_id is not None:
        config.DATASET.SUBJECT_ID = args.subject_id
        config_dict["DATASET"]["SUBJECT_ID"] = args.subject_id

    if args.project is not None:
        config.SETTINGS.PROJECT_NAME = args.project
        config_dict["SETTINGS"]["PROJECT_NAME"] = args.project

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
                        "subject_id": str(config.DATASET.SUBJECT_ID),
                        "script_name": "phase1_generation.py",
                        "phase1_only": "true",
                        "progressive_hash_unlock": str(args.progressive_hash_unlock),
                    },
                )

                active_run = mlflow.active_run()
                mlflow_run_id = active_run.info.run_id if active_run else None

                # Log parameters
                mlflow_params = {
                    "unique_id": unique_id,
                    "subject_id": config.DATASET.SUBJECT_ID,
                    "learning_rate": config.TRAINING.LR,
                    "batch_size": config.TRAINING.BATCH_SIZE,
                    "initial_gen_epochs": config.TRAINING.initial_generation_epochs,
                    "hash_n_levels": args.hash_n_levels,
                    "hash_n_features_per_level": args.hash_n_features_per_level,
                    "hash_log2_size": args.hash_log2_size,
                    "hash_base_resolution": args.hash_base_resolution,
                    "hash_per_level_scale": args.hash_per_level_scale,
                    "concat_coords": args.concat_coords,
                    "progressive_hash_unlock": args.progressive_hash_unlock,
                    "hash_unlock_end_fraction": args.hash_unlock_end_fraction,
                    "loss_type": config.TRAINING.LOSS,
                    "disable_early_stopping": args.disable_early_stopping,
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
    exp_base_dir = f"runs/{unique_id}"
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
    weight_subdir = f"{hash_tag}_{proj_short}_w{es_suffix}"
    image_subdir = f"{hash_tag}_{proj_short}_img{es_suffix}"

    weight_dir = os.path.join(exp_base_dir, weight_subdir)
    image_dir = os.path.join(exp_base_dir, image_subdir)

    pathlib.Path(weight_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)

    # Log experiment details to central CSV
    csv_fieldnames = [
        "unique_id",
        "timestamp",
        "script_name",
        "subject_id",
        "lr_contrast1",
        "lr_contrast2",
        "learning_rate",
        "batch_size",
        "initial_gen_epochs",
        "hash_n_levels",
        "hash_n_features_per_level",
        "hash_log2_size",
        "hash_base_resolution",
        "hash_per_level_scale",
        "concat_coords",
        "progressive_hash_unlock",
        "hash_unlock_end_fraction",
        "loss_type",
        "mlflow_run_id",
        "mlflow_tracking_uri",
        "mlflow_experiment",
        "weight_dir",
        "image_dir",
        "config_path",
        "seed",
        "early_stopping",
        "device",
        "notes",
    ]

    exp_log_data = {
        "unique_id": unique_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script_name": "phase1_generation.py",
        "subject_id": config.DATASET.SUBJECT_ID,
        "lr_contrast1": config.DATASET.LR_CONTRAST1,
        "lr_contrast2": config.DATASET.LR_CONTRAST2,
        "learning_rate": config.TRAINING.LR,
        "batch_size": config.TRAINING.BATCH_SIZE,
        "initial_gen_epochs": config.TRAINING.initial_generation_epochs,
        "hash_n_levels": args.hash_n_levels,
        "hash_n_features_per_level": args.hash_n_features_per_level,
        "hash_log2_size": args.hash_log2_size,
        "hash_base_resolution": args.hash_base_resolution,
        "hash_per_level_scale": args.hash_per_level_scale,
        "concat_coords": args.concat_coords,
        "progressive_hash_unlock": args.progressive_hash_unlock,
        "hash_unlock_end_fraction": args.hash_unlock_end_fraction,
        "loss_type": config.TRAINING.LOSS,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "mlflow_experiment": mlflow_experiment_name,
        "weight_dir": weight_dir,
        "image_dir": image_dir,
        "config_path": args.config,
        "seed": config.TRAINING.SEED,
        "early_stopping": config.TRAINING.EARLY_STOPPING.ENABLED
        if hasattr(config.TRAINING, "EARLY_STOPPING")
        else False,
        "device": str(device),
        "notes": f"phase1_only_hash_grid_model_{config.GENERATION_MODEL.TYPE}_freenerf",
    }

    pathlib.Path("runs").mkdir(exist_ok=True)
    csv_path = "runs/experiment_log_phase1_freenerf.csv"
    if not safe_append_to_csv(csv_path, exp_log_data, csv_fieldnames):
        print(f"Warning: Could not write to central CSV, saving metadata to {exp_base_dir}/metadata.json")
        with open(os.path.join(exp_base_dir, "metadata.json"), "w") as f:
            json.dump(exp_log_data, f, indent=2)

    # Save full config snapshot
    config_snapshot = {
        "unique_id": unique_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "config": config_dict,
        "mlflow_info": {
            "run_id": mlflow_run_id,
            "tracking_uri": mlflow_tracking_uri,
            "experiment": mlflow_experiment_name,
        },
    }
    with open(os.path.join(exp_base_dir, "config_snapshot.json"), "w") as f:
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
        input_range=(-1, 1),  # Match existing coordinate normalization range
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
        model = networks.Siren(
            [3, 1024, 1024, 1024, 1024, 1], True, 15
        )  # following the INR setting for now
    else:
        raise ValueError(f"Unknown generation model type: {config.GENERATION_MODEL.TYPE}")

    # Initialize loss function
    if config.TRAINING.LOSS == "L1Loss":
        criterion = nn.L1Loss()
    elif config.TRAINING.LOSS == "MSELoss":
        criterion = nn.MSELoss()
    else:
        raise ValueError("Loss function not defined!")

    # Build compact model name
    gen_tag = "mlp"
    if config.GENERATION_MODEL.TYPE == "siren":
        gen_tag = "siren"
    elif config.GENERATION_MODEL.TYPE not in ("mlp",):
        gen_tag = str(config.GENERATION_MODEL.TYPE)

    loss_tag = build_loss_tag(config)
    model_name = (
        f"{config.DATASET.SUBJECT_ID}_"
        f"proj-{proj_short}_"
        f"{hash_tag}_"
        f"{gen_tag}_"
        f"{loss_tag}_"
        f"b{config.TRAINING.BATCH_SIZE}"
    )

    # Initialize optimizer
    if config.TRAINING.OPTIM == "Adam":
        params_img = list(model.parameters())
        params_img += list(input_mapper.parameters())  # Add hash encoder parameters

        optimizer_img = torch.optim.AdamW(
            params_img, lr=config.TRAINING.LR * 3, weight_decay=5e-5
        )
    else:
        raise ValueError("Optim not defined!")

    # Initialize scheduler (T_max matches full script for consistent LR evolution)
    generation_epochs = getattr(config.TRAINING, "generation_epochs", 0)
    scheduler_img = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer_img,
        T_max=config.TRAINING.initial_generation_epochs + generation_epochs,
    )

    # Print parameter counts
    print(f"Number of MLP parameters {sum(p.numel() for p in model.parameters())}")
    print(f"Number of hash encoder parameters {sum(p.numel() for p in input_mapper.parameters())}")

    # Load training data
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.TRAINING.BATCH_SIZE,
        shuffle=config.TRAINING.SHUFFLING,
        num_workers=config.SETTINGS.NUM_WORKERS,
    )

    # Move model to device
    model = model.to(device)

    ################ TRAINING for image generation #######################
    epoch_rough_image_generation = config.TRAINING.initial_generation_epochs

    # Initialize early stopping for image generation phase
    early_stopping_enabled = (
        hasattr(config.TRAINING, "EARLY_STOPPING")
        and config.TRAINING.EARLY_STOPPING.ENABLED
    )
    if early_stopping_enabled:
        # Get delta_fraction from config (used for improvement threshold)
        es_delta_fraction = getattr(
            config.TRAINING.EARLY_STOPPING, "DELTA_FRACTION",
            config.TRAINING.EARLY_STOPPING.DELTA  # fallback to DELTA if DELTA_FRACTION not set
        )
        early_stop_img = EarlyStopping(
            patience=config.TRAINING.EARLY_STOPPING.PATIENCE,
            delta_fraction=es_delta_fraction,
            mode="min",
            verbose=config.TRAINING.EARLY_STOPPING.VERBOSE,
        )
        # Track best weights in memory (to save only once at early stop)
        best_model_state = None
        best_hash_state = None
        best_epoch = None

    # Track the actual saved checkpoint (may differ from final epoch if early stopped)
    saved_checkpoint_epoch = epoch_rough_image_generation - 1  # default to final epoch

    # Compute total steps for progressive unlock
    total_steps = epoch_rough_image_generation * len(train_dataloader)

    print("=" * 60)
    print("Starting Phase 1 Image Generation Training...")
    print(f"  Epochs: {epoch_rough_image_generation}")
    print(f"  Batch size: {config.TRAINING.BATCH_SIZE}")
    print(f"  Learning rate: {config.TRAINING.LR * 3}")
    print(f"  Loss: {config.TRAINING.LOSS}")
    if args.progressive_hash_unlock:
        print(f"  Progressive hash unlock: enabled (end fraction: {args.hash_unlock_end_fraction})")
    print("=" * 60)

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
            input_mapper.set_training_progress(current_step / total_steps)

            batch_metrics = {}

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

            if args.logging:
                img_step = epoch * len(train_dataloader) + batch_idx
                batch_metrics.update({"img_loss": loss_mse.item()})

                # Log progressive hash unlock metrics
                if args.progressive_hash_unlock:
                    level_weights = input_mapper.compute_level_weights()
                    active_levels = (level_weights > 0.5).sum().item()
                    batch_metrics.update({
                        "hash_unlock_progress": current_step / total_steps,
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
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_hash_state = {k: v.cpu().clone() for k, v in input_mapper.state_dict().items()}

            if should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"  Best loss was at epoch {best_epoch} (loss: {early_stop_img.best_loss:.6f})")
                if args.logging:
                    mlflow_log_metrics(mlflow, {"img_early_stop_epoch": epoch}, step=epoch)
                    mlflow_log_metrics(mlflow, {"img_best_epoch": best_epoch}, step=epoch)

                # Save the BEST weights (not current epoch's weights)
                saved_checkpoint_epoch = best_epoch
                best_model_path = os.path.join(weight_dir, f"{model_name}_e{best_epoch}_model.pt")
                torch.save(best_model_state, best_model_path)
                torch.save(best_hash_state, hash_ckpt_for(best_model_path))

                # Load best weights back into models for inference
                model.load_state_dict(best_model_state)
                input_mapper.load_state_dict(best_hash_state)
                break

        # Save checkpoint at final epoch (only if we didn't early stop)
        if epoch == (epoch_rough_image_generation - 1):
            torch.save(model.state_dict(), model_path)
            torch.save(input_mapper.state_dict(), hash_ckpt_for(model_path))
            saved_checkpoint_epoch = epoch

    ################ INFERENCE for image generation #######################
    print("=" * 60)
    print("Running Phase 1 Inference...")
    print("=" * 60)

    infer_batch_size = 10000

    (out_affine1, out_loader1, out_image1, out_dim_xyz1) = get_image_frame(
        dataset, config, batch_size=infer_batch_size, input_idx=1
    )

    model.eval()
    input_mapper.eval()

    # Use the actual saved checkpoint epoch for naming
    saved_model_name_epoch = f"{model_name}_e{saved_checkpoint_epoch}_model.pt"
    init_generation_fname = os.path.join(
        image_dir,
        saved_model_name_epoch.replace("model.pt", f"_ct1_init.nii.gz"),
    )

    if not os.path.exists(init_generation_fname):
        for batch_idx, (data) in enumerate(out_loader1):
            data = data.to(device)
            data = input_mapper(data)

            output = model(data)

            out_image1[
                batch_idx * infer_batch_size : (
                    batch_idx * infer_batch_size + len(output)
                ),
                :,
            ] = output.cpu().detach().numpy()

        print("Generating NIFTIs for contrast1")
        (x_dim, y_dim, z_dim) = out_dim_xyz1
        scaler = MinMaxScaler()
        label_arr = np.array(out_image1, dtype=np.float32)
        model_intensities_contrast1 = scaler.fit_transform(label_arr.reshape(-1, 1))
        img_contrast1 = model_intensities_contrast1.reshape(
            (x_dim, y_dim, z_dim)
        )
        img = nib.Nifti1Image(img_contrast1, out_affine1)
        nib.save(img, init_generation_fname)
        print(f"Saved Phase 1 output to: {init_generation_fname}")

    # Finish MLflow logging session
    if args.logging and mlflow is not None and mlflow.active_run() is not None:
        mlflow.end_run()

    print("=" * 60)
    print("Phase 1 Image Generation Training Complete!")
    print(f"Experiment ID: {unique_id}")
    print(f"Output directory: {exp_base_dir}")
    print(f"Weights directory: {weight_dir}")
    print(f"Images directory: {image_dir}")
    print(f"Saved checkpoint: {saved_model_name_epoch}")
    print("")
    print("To run Phase 2 registration with this checkpoint:")
    print(f"  python phase2_registration.py --config {args.config} \\")
    print(f"      --phase1_checkpoint {os.path.join(weight_dir, saved_model_name_epoch)} \\")
    print(f"      --registration_epochs 100 {'--concat_coords ' if args.concat_coords else ''}--logging")
    print("=" * 60)


def _cli():
    main(parse_args())


if __name__ == "__main__":
    _cli()
