"""
Hash encoder wrapper with FreeNeRF-style progressive hash unlock.

Progressive unlock uses per-level weights `w[i] = clamp(visible - i, 0, 1)`,
where `visible` grows with training progress so higher-frequency levels are
introduced gradually during training.
"""

import torch
import torch.nn as nn

from sims_mri.hash_embeddings import HashEmbedder


class HashEncoderWrapper(nn.Module):
    """Wraps HashEmbedder with coordinate normalization, optional coord concat, and progressive unlock."""

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        per_level_scale: float = 1.39,
        concat_coords: bool = True,
        input_range: tuple = (-1, 1),
        progressive_unlock: bool = False,
        unlock_end_fraction: float = 0.9,
    ):
        """
        Args:
            n_levels: Number of resolution levels in the hash grid
            n_features_per_level: Number of features per level
            log2_hashmap_size: Log2 of hash table size (e.g., 19 -> 2^19 entries)
            base_resolution: Base resolution at coarsest level
            per_level_scale: Resolution multiplier between levels
            concat_coords: If True, concatenate original coordinates with embeddings
            input_range: Expected input coordinate range, will be normalized to [0, 1]
            progressive_unlock: If True, enable FreeNeRF-style progressive hash level unlock
            unlock_end_fraction: Fraction of training at which all hash levels are unlocked
        """
        super().__init__()

        self.hash_embedder = HashEmbedder(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=per_level_scale,
        )

        self.concat_coords = concat_coords
        self.input_min = input_range[0]
        self.input_max = input_range[1]
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.progressive_unlock = progressive_unlock
        self.unlock_end_fraction = unlock_end_fraction
        self._current_progress = 0.0
        embed_dim = n_levels * n_features_per_level
        self.output_dim = embed_dim + 3 if concat_coords else embed_dim

    def set_training_progress(self, progress: float):
        self._current_progress = min(max(progress, 0.0), 1.0)

    def compute_level_weights(self) -> torch.Tensor:
        """Compute train-time per-level weights from training progress.

        Returns tensor of shape [n_levels] with weights in [0, 1].
        """
        if not self.progressive_unlock or not self.training:
            return torch.ones(self.n_levels)

        unlock_progress = min(self._current_progress / self.unlock_end_fraction, 1.0)
        visible = unlock_progress * self.n_levels

        idx = torch.arange(self.n_levels, dtype=torch.float32)
        weights = torch.clamp(visible - idx, 0.0, 1.0)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hash encoder.

        Args:
            x: Input coordinates of shape [B, 3] in input_range

        Returns:
            Coordinates are normalized to [0, 1] and clamped so slightly
            out-of-range points do not index invalid hash cells.
            Hash embeddings (optionally concatenated with coords) of shape:
            - [B, n_levels * n_features_per_level + 3] if concat_coords=True
            - [B, n_levels * n_features_per_level] if concat_coords=False
        """
        x_norm = (x - self.input_min) / (self.input_max - self.input_min)
        x_norm = torch.clamp(x_norm, 0, 1)
        embeddings = self.hash_embedder(x_norm)

        if self.progressive_unlock and self.training:
            level_weights = self.compute_level_weights().to(embeddings.device)
            B = embeddings.shape[0]
            embeddings = embeddings.view(B, self.n_levels, self.n_features_per_level)
            embeddings = embeddings * level_weights.view(1, -1, 1)
            embeddings = embeddings.view(B, -1)

        if self.concat_coords:
            return torch.cat([embeddings, x], dim=-1)
        return embeddings

    def get_config(self) -> dict:
        """Return configuration for logging/checkpointing."""
        return {
            "n_levels": self.n_levels,
            "n_features_per_level": self.n_features_per_level,
            "concat_coords": self.concat_coords,
            "input_range": (self.input_min, self.input_max),
            "output_dim": self.output_dim,
            "progressive_unlock": self.progressive_unlock,
            "unlock_end_fraction": self.unlock_end_fraction,
        }
