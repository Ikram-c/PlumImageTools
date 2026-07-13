from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoencoderSettings:
    encoding_dim: int = 8
    architecture: str = "deep"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    weight_decay: float = 1e-5
    beta_vae: float = 0.1
    sparsity_weight: float = 0.01
    early_stop_patience: int = 10
    early_stop_min_epoch: int = 20


@dataclass(frozen=True)
class ClusteringSettings:
    max_clusters: int = 8
    dbscan_min_samples: int = 5
    dbscan_eps_min: float = 0.1
    dbscan_eps_max: float = 2.0
    dbscan_eps_steps: int = 10
    random_state: int = 42
    compare_with_raw: bool = True