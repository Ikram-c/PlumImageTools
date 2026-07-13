from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from vision_toolkit.clustering.autoencoder import AutoencoderTrainer
from vision_toolkit.clustering.feature_extraction import extract_combined_features
from vision_toolkit.clustering.settings import (
    AutoencoderSettings,
    ClusteringSettings,
)

logger = logging.getLogger(__name__)

_META_COLUMNS = ("category_id", "annotation_id", "image_id")


@dataclass
class ClusteringResult:
    algorithm: str
    cluster_labels: np.ndarray
    feature_names: list[str]
    feature_importance: dict[str, float]
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    n_clusters: int
    cluster_centers: Optional[np.ndarray] = None
    feature_data: Optional[pd.DataFrame] = None
    encoded_features: Optional[np.ndarray] = None
    reconstruction_error: Optional[float] = None


class EnhancedClusteringAnalyzer:

    def __init__(
        self,
        feature_data: pd.DataFrame,
        use_autoencoder: bool = True,
        autoencoder_settings: AutoencoderSettings = AutoencoderSettings(),
        clustering_settings: ClusteringSettings = ClusteringSettings(),
    ):
        self.feature_data = feature_data.copy()
        self.use_autoencoder = use_autoencoder
        self.ae_settings = autoencoder_settings
        self.settings = clustering_settings
        self.scaler = StandardScaler()
        self.results: dict[str, ClusteringResult] = {}
        self.best_result: Optional[ClusteringResult] = None
        self.autoencoder: Optional[AutoencoderTrainer] = None
        self.X_encoded: Optional[np.ndarray] = None
        self.reconstruction_error: Optional[float] = None

        self.feature_columns = [
            c for c in feature_data.columns if c not in _META_COLUMNS
        ]
        self.X = self.feature_data[self.feature_columns].fillna(0)
        self.X_scaled = self.scaler.fit_transform(self.X)

        if self.use_autoencoder:
            self._train_autoencoder()

    def _train_autoencoder(self) -> None:
        logger.info("Training autoencoder for feature learning...")
        self.autoencoder = AutoencoderTrainer(
            input_dim=len(self.feature_columns), settings=self.ae_settings
        )
        self.autoencoder.train(self.X_scaled)
        self.X_encoded = self.autoencoder.encode(self.X_scaled)
        _, self.reconstruction_error = self.autoencoder.reconstruct(self.X_scaled)
        logger.info(
            "Autoencoder trained. Reconstruction error: %.6f. Dims %d -> %d",
            self.reconstruction_error,
            len(self.feature_columns),
            self.X_encoded.shape[1],
        )

    def find_optimal_clusters(
        self, algorithms: Optional[list[str]] = None
    ) -> dict[str, ClusteringResult]:
        if algorithms is None:
            algorithms = ["kmeans", "gaussian_mixture", "hierarchical"]

        X_for_clustering = (
            self.X_encoded if self.use_autoencoder else self.X_scaled
        )
        suffix = "_encoded" if self.use_autoencoder else "_raw"

        for algorithm in algorithms:
            result = self._optimise_algorithm(algorithm, X_for_clustering)
            if result is not None:
                self.results[algorithm + suffix] = result

        dbscan_result = self._optimise_dbscan(X_for_clustering)
        if dbscan_result is not None:
            self.results["dbscan" + suffix] = dbscan_result

        if self.use_autoencoder and self.settings.compare_with_raw:
            raw_result = self._optimise_algorithm("kmeans", self.X_scaled)
            if raw_result is not None:
                self.results["kmeans_raw_comparison"] = raw_result

        self.best_result = max(
            self.results.values(), key=lambda r: r.silhouette_score
        )
        logger.info(
            "Best clustering: %s (silhouette %.3f)",
            self.best_result.algorithm,
            self.best_result.silhouette_score,
        )
        return self.results

    def _optimise_algorithm(
        self, algorithm: str, X: np.ndarray
    ) -> Optional[ClusteringResult]:
        best_score = -1.0
        best_n_clusters = 2
        for n_clusters in tqdm(
            range(2, self.settings.max_clusters + 1),
            desc=f"{algorithm} optimization",
        ):
            try:
                labels = self._cluster_with_algorithm(algorithm, n_clusters, X)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            except Exception as e:
                logger.debug("%s n=%d failed: %s", algorithm, n_clusters, e)
                continue
        if best_score < 0:
            return None
        final_labels = self._cluster_with_algorithm(algorithm, best_n_clusters, X)
        return self._analyze_clustering_result(
            algorithm, final_labels, best_n_clusters, X
        )

    def _optimise_dbscan(self, X: np.ndarray) -> Optional[ClusteringResult]:
        eps_values = np.linspace(
            self.settings.dbscan_eps_min,
            self.settings.dbscan_eps_max,
            self.settings.dbscan_eps_steps,
        )
        best_score = -1.0
        best_labels = None
        for eps in tqdm(eps_values, desc="DBSCAN optimization"):
            try:
                labels = DBSCAN(
                    eps=eps, min_samples=self.settings.dbscan_min_samples
                ).fit_predict(X)
                if len(np.unique(labels)) > 1 and -1 not in labels:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score = score
                        best_labels = labels
            except Exception as e:
                logger.debug("dbscan eps=%.2f failed: %s", eps, e)
                continue
        if best_labels is None:
            return None
        return self._analyze_clustering_result(
            "dbscan", best_labels, len(np.unique(best_labels)), X
        )

    def _cluster_with_algorithm(
        self, algorithm: str, n_clusters: int, X: np.ndarray
    ) -> np.ndarray:
        seed = self.settings.random_state
        clusterers = {
            "kmeans": lambda: KMeans(
                n_clusters=n_clusters, random_state=seed, n_init=10
            ),
            "gaussian_mixture": lambda: GaussianMixture(
                n_components=n_clusters, random_state=seed
            ),
            "hierarchical": lambda: AgglomerativeClustering(
                n_clusters=n_clusters
            ),
        }
        if algorithm not in clusterers:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        return clusterers[algorithm]().fit_predict(X)

    def _analyze_clustering_result(
        self, algorithm: str, labels: np.ndarray, n_clusters: int, X: np.ndarray
    ) -> ClusteringResult:
        multi = len(np.unique(labels)) > 1
        sil = silhouette_score(X, labels) if multi else 0.0
        ch = calinski_harabasz_score(X, labels) if multi else 0.0
        db = davies_bouldin_score(X, labels) if multi else float("inf")

        is_encoded_space = (
            self.use_autoencoder
            and self.X_encoded is not None
            and X.shape[1] == self.X_encoded.shape[1]
        )
        if is_encoded_space:
            importance = self._calculate_encoded_feature_importance(labels)
        else:
            importance = self._calculate_feature_importance(labels, X)

        cluster_centers = None
        if algorithm in ("kmeans", "gaussian_mixture"):
            unique_labels = np.unique(labels)
            cluster_centers = np.array(
                [X[labels == label].mean(axis=0) for label in unique_labels]
            )

        feature_df = self.feature_data.copy()
        feature_df["cluster_label"] = labels

        return ClusteringResult(
            algorithm=algorithm,
            cluster_labels=labels,
            feature_names=self.feature_columns,
            feature_importance=importance,
            silhouette_score=sil,
            calinski_harabasz_score=ch,
            davies_bouldin_score=db,
            n_clusters=n_clusters,
            cluster_centers=cluster_centers,
            feature_data=feature_df,
            encoded_features=self.X_encoded if self.use_autoencoder else None,
            reconstruction_error=self.reconstruction_error,
        )

    def _calculate_encoded_feature_importance(
        self, labels: np.ndarray
    ) -> dict[str, float]:
        encoder_weights = None
        if hasattr(self.autoencoder.model, "encoder"):
            first_layer = list(self.autoencoder.model.encoder.children())[0]
            if isinstance(first_layer, nn.Linear):
                encoder_weights = first_layer.weight.data.cpu().numpy()

        encoded_names = [f"encoded_{i}" for i in range(self.X_encoded.shape[1])]
        encoded_importance = self._calculate_feature_importance(
            labels, self.X_encoded, feature_names=encoded_names
        )

        if encoder_weights is None:
            return self._calculate_feature_importance(labels, self.X_scaled)

        final_importance = {}
        for i, feature_name in enumerate(self.feature_columns):
            importance_sum = 0.0
            for key, value in encoded_importance.items():
                if value <= 0:
                    continue
                encoded_idx = int(key.split("_")[1])
                if encoded_idx < encoder_weights.shape[0]:
                    importance_sum += abs(encoder_weights[encoded_idx, i]) * value
            final_importance[feature_name] = importance_sum
        return final_importance

    def _calculate_feature_importance(
        self,
        labels: np.ndarray,
        X: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> dict[str, float]:
        if feature_names is None:
            feature_names = self.feature_columns

        scores: dict[str, list[float]] = {name: [] for name in feature_names}

        try:
            mi = mutual_info_classif(
                X, labels, random_state=self.settings.random_state
            )
            for i, name in enumerate(feature_names):
                scores[name].append(mi[i])
        except Exception as e:
            logger.debug("Mutual info failed: %s", e)

        try:
            rf = self._build_adaptive_forest(labels, len(feature_names))
            rf.fit(X, labels)
            for i, name in enumerate(feature_names):
                scores[name].append(rf.feature_importances_[i])
        except Exception as e:
            logger.debug("Random forest importance failed: %s", e)

        try:
            cluster_means = np.array(
                [X[labels == label].mean(axis=0) for label in np.unique(labels)]
            )
            variances = np.var(cluster_means, axis=0)
            for i, name in enumerate(feature_names):
                scores[name].append(variances[i])
        except Exception as e:
            logger.debug("Cluster variance importance failed: %s", e)

        return {
            name: float(np.mean(vals)) if vals else 0.0
            for name, vals in scores.items()
        }

    def _build_adaptive_forest(
        self, labels: np.ndarray, n_features: int
    ) -> RandomForestClassifier:
        n_classes = len(np.unique(labels))
        n_samples = len(labels)
        return RandomForestClassifier(
            n_estimators=max(50, min(200, n_classes * 15)),
            max_depth=max(3, min(15, int(np.log2(n_classes)) + 3)),
            min_samples_split=max(2, min(20, n_samples // (n_classes * 10))),
            min_samples_leaf=max(1, min(10, n_samples // (n_classes * 20))),
            max_features=min(n_features, max(1, int(np.sqrt(n_features)))),
            random_state=self.settings.random_state,
            n_jobs=-1,
        )

    def plot_clustering_results(
        self,
        result: Optional[ClusteringResult] = None,
        save_path: Optional[str] = None,
    ) -> None:
        if result is None:
            result = self.best_result
        X_viz = (
            result.encoded_features
            if result.encoded_features is not None
            else self.X_scaled
        )

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"Clustering Analysis: {result.algorithm.upper()}")

        tsne = TSNE(
            n_components=2,
            random_state=self.settings.random_state,
            perplexity=min(30, max(2, len(X_viz) // 4)),
        )
        X_tsne = tsne.fit_transform(X_viz)
        scatter = axes[0].scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=result.cluster_labels,
            cmap="viridis",
            alpha=0.7,
        )
        axes[0].set_title("t-SNE Visualization of Clusters")
        plt.colorbar(scatter, ax=axes[0])

        importance_items = sorted(
            result.feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:12]
        if importance_items:
            names, values = zip(*importance_items)
            axes[1].barh(range(len(names)), values)
            axes[1].set_yticks(range(len(names)))
            axes[1].set_yticklabels(
                [n.replace("_", " ") for n in names], fontsize=8
            )
        axes[1].set_title("Top Features Separating Clusters")

        unique_labels, counts = np.unique(
            result.cluster_labels, return_counts=True
        )
        axes[2].bar(unique_labels, counts, alpha=0.7)
        axes[2].set_title("Cluster Size Distribution")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Clustering visualization saved to %s", save_path)
        plt.close(fig)

    def generate_report(self, save_path: Optional[str] = None) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("ANNOTATION CLUSTERING ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Total annotations analyzed: {len(self.feature_data)}")
        lines.append(f"Original features: {len(self.feature_columns)}")
        lines.append(f"Autoencoder used: {self.use_autoencoder}")
        if self.use_autoencoder:
            lines.append(f"Encoded dims: {self.X_encoded.shape[1]}")
            lines.append(
                f"Reconstruction error: {self.reconstruction_error:.6f}"
            )
        lines.append("")

        if self.best_result is not None:
            best = self.best_result
            lines.append("BEST CLUSTERING RESULT")
            lines.append("-" * 40)
            lines.append(f"Algorithm: {best.algorithm.upper()}")
            lines.append(f"Clusters: {best.n_clusters}")
            lines.append(f"Silhouette: {best.silhouette_score:.4f}")
            lines.append(
                f"Calinski-Harabasz: {best.calinski_harabasz_score:.4f}"
            )
            lines.append(f"Davies-Bouldin: {best.davies_bouldin_score:.4f}")
            lines.append("")
            lines.append("TOP 10 SEPARATING FEATURES")
            lines.append("-" * 40)
            top = sorted(
                best.feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]
            for rank, (feature, score) in enumerate(top, 1):
                lines.append(f"{rank:2d}. {feature:<30} {score:.6f}")
            lines.append("")

        lines.append("ALL RESULTS")
        lines.append("-" * 40)
        for name, result in sorted(
            self.results.items(),
            key=lambda kv: kv[1].silhouette_score,
            reverse=True,
        ):
            lines.append(
                f"{name:<28} | clusters: {result.n_clusters:2d} "
                f"| silhouette: {result.silhouette_score:.4f}"
            )

        report_text = "\n".join(lines)
        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)
            logger.info("Report saved to %s", save_path)
        return report_text


def run_clustering_analysis(
    json_path: str,
    output_dir: str = "clustering_results",
    use_autoencoder: bool = True,
    autoencoder_settings: AutoencoderSettings = AutoencoderSettings(),
    clustering_settings: ClusteringSettings = ClusteringSettings(),
) -> tuple[EnhancedClusteringAnalyzer, dict[str, ClusteringResult]]:
    os.makedirs(output_dir, exist_ok=True)

    combined = extract_combined_features(json_path)
    logger.info(
        "Extracted %d features for %d annotations",
        len(combined.columns) - 3,
        len(combined),
    )

    analyzer = EnhancedClusteringAnalyzer(
        combined,
        use_autoencoder=use_autoencoder,
        autoencoder_settings=autoencoder_settings,
        clustering_settings=clustering_settings,
    )
    results = analyzer.find_optimal_clusters()

    analyzer.plot_clustering_results(
        save_path=os.path.join(output_dir, "clustering_visualization.png")
    )
    analyzer.generate_report(
        save_path=os.path.join(output_dir, "clustering_report.txt")
    )
    analyzer.best_result.feature_data.to_csv(
        os.path.join(output_dir, "clustered_annotations.csv"), index=False
    )
    if use_autoencoder and analyzer.autoencoder is not None:
        torch.save(
            analyzer.autoencoder.model.state_dict(),
            os.path.join(output_dir, "autoencoder_model.pth"),
        )
    return analyzer, results