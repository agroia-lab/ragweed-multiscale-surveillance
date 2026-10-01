#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Instituto de Investigaciones Agropecuarias (INIA), Chile
"""
Embedding backbone components used by embed_ortho_tiles.py
==========================================================

Verbatim subset of the original image-embedding module used in the
preprint workflow: the CNN feature extractor (ImageNet-pretrained
ResNet50 / EfficientNet with global average pooling), the generic image
dataset, batched embedding extraction, and PCA/UMAP/t-SNE reduction.
Only the components imported by embed_ortho_tiles.py are included; the
original module's project-specific configuration and entry point are not.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ============================================================================
# FEATURE EXTRACTOR MODEL
# ============================================================================

class FeatureExtractor(nn.Module):
    """
    Extract features from images using pre-trained CNN backbone.
    Uses global average pooling to get fixed-size embeddings.
    """

    def __init__(self, model_name: str = "resnet50"):
        super().__init__()

        if model_name == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            # Remove the final FC layer
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            self.embedding_dim = 2048

        elif model_name == "efficientnet_b0":
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            self.embedding_dim = 1280

        elif model_name == "efficientnet_b2":
            base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(base_model.children())[:-1])
            self.embedding_dim = 1408

        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = self.pool(features)
        return features.flatten(1)


# ============================================================================
# IMAGE DATASET
# ============================================================================

class ImageDataset(Dataset):
    """Dataset for loading images from directory structure."""

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        max_images: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Find all images
        extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        self.image_paths = []

        print(f"Scanning for images in {root_dir}...")
        for ext in extensions:
            self.image_paths.extend(list(self.root_dir.rglob(f"*{ext}")))

        # Sort for reproducibility
        self.image_paths.sort()

        # Limit if specified
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        print(f"Found {len(self.image_paths)} images")

        # Extract metadata from paths
        self.metadata = self._extract_metadata()

    def _extract_metadata(self) -> List[Dict]:
        """Extract location/source info from path structure."""
        metadata = []
        for path in self.image_paths:
            rel_path = path.relative_to(self.root_dir)
            parts = rel_path.parts

            # Extract location (first directory level)
            location = parts[0] if len(parts) > 0 else "unknown"

            # Extract source type (Celular, Dron, etc.)
            source_type = "unknown"
            for part in parts:
                if "Celular" in part:
                    source_type = "Celular"
                    break
                elif "Dron" in part or "dron" in part:
                    source_type = "Dron"
                    break
                elif "MEDIA" in part:
                    source_type = "Camera"
                    break

            metadata.append({
                "filename": path.name,
                "full_path": str(path),
                "relative_path": str(rel_path),
                "location": location,
                "source_type": source_type,
            })

        return metadata

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, idx

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image on error
            if self.transform:
                return torch.zeros(3, 224, 224), idx
            return Image.new('RGB', (224, 224)), idx


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def extract_embeddings(
    dataset: ImageDataset,
    model: FeatureExtractor,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda"
) -> np.ndarray:
    """
    Extract embeddings for all images in dataset.

    Returns:
        embeddings: numpy array of shape (n_images, embedding_dim)
    """
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )

    embeddings = []

    print(f"Extracting embeddings using {device}...")
    with torch.no_grad():
        for images, indices in tqdm(dataloader, desc="Processing batches"):
            images = images.to(device)
            features = model(images)
            embeddings.append(features.cpu().numpy())

    return np.vstack(embeddings)


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    **kwargs
) -> np.ndarray:
    """
    Apply dimensionality reduction to embeddings.

    Args:
        embeddings: Input embeddings (n_samples, n_features)
        method: "umap", "tsne", or "pca"
        n_components: Output dimensions (2 or 3)
        **kwargs: Additional parameters for the method

    Returns:
        reduced: Reduced coordinates (n_samples, n_components)
    """
    print(f"Applying {method.upper()} to reduce to {n_components}D...")

    # Apply PCA first for UMAP/t-SNE if embeddings are high-dimensional
    if method in ["umap", "tsne"] and embeddings.shape[1] > 50:
        from sklearn.decomposition import PCA
        pca_components = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
        print(f"  Pre-reducing with PCA to {pca_components} dimensions...")
        pca = PCA(n_components=pca_components, random_state=42)
        embeddings = pca.fit_transform(embeddings)
        print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    if method == "umap":
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
            metric="cosine",
            random_state=42,
            verbose=True
        )
        reduced = reducer.fit_transform(embeddings)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_components,
            perplexity=kwargs.get("perplexity", 30),
            random_state=42,
            verbose=1,
            max_iter=1000
        )
        reduced = reducer.fit_transform(embeddings)

    elif method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        print(f"  PCA explained variance: {reducer.explained_variance_ratio_.sum():.2%}")

    else:
        raise ValueError(f"Unknown method: {method}")

    return reduced
