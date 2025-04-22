from datasets import load_dataset, Dataset, DatasetDict
from transformers import SegformerImageProcessor
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from PIL import Image
import random
import logging
import torch

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# Constants and label maps
# ────────────────────────────────────────────────────────────────────────────────
PASCAL_VOC_LABEL_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
PASCAL_VOC_ID2LABEL = {i: label for i, label in enumerate(PASCAL_VOC_LABEL_NAMES)}
PASCAL_VOC_LABEL2ID = {label: i for i, label in PASCAL_VOC_ID2LABEL.items()}
NUM_PASCAL_VOC_LABELS = len(PASCAL_VOC_ID2LABEL)

# Pixels with this value will be ignored by Cross‑Entropy
PASCAL_VOC_IGNORE_INDEX = 255


# ────────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ────────────────────────────────────────────────────────────────────────────────
def load_pascal_voc_dataset(
    dataset_name: str = "nateraw/pascal-voc-2012",
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load the Pascal‑VOC 2012 dataset from Hugging Face Datasets.

    Returns a DatasetDict with splits:
      - "train"
      - "test"  (former HF "validation")
    """
    logger.info(f"Loading dataset '{dataset_name}'…")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    # HF version has "train"/"val"; rename "val"→"test" for clarity
    if "train" not in dataset or "validation" not in dataset:
        if "val" in dataset:
            dataset["validation"] = dataset.pop("val")
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' must have 'train' and 'validation' splits."
            )

    dataset["test"] = dataset.pop("validation")
    logger.info(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    return dataset


def create_train_val_test_splits(
    full_dataset: Dataset,
    val_percentage: float = 0.1,
    test_percentage: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a single HF Dataset into train/val/test subsets.
    """
    num_samples = len(full_dataset)
    if val_percentage + test_percentage >= 1.0:
        raise ValueError("Sum of val_percentage and test_percentage must be < 1.0")

    indices = list(range(num_samples))
    random.seed(seed)
    random.shuffle(indices)

    num_test = int(num_samples * test_percentage)
    num_val  = int(num_samples * val_percentage)

    test_indices = indices[:num_test]
    val_indices  = indices[num_test : num_test + num_val]
    train_indices = indices[num_test + num_val:]

    train_dataset = full_dataset.select(train_indices)
    val_dataset   = full_dataset.select(val_indices)
    test_dataset  = full_dataset.select(test_indices)

    logger.info(
        f"Split dataset → Train={len(train_dataset)}, "
        f"Val={len(val_dataset)}, Test={len(test_dataset)}"
    )
    return train_dataset, val_dataset, test_dataset


# ────────────────────────────────────────────────────────────────────────────────
# Pre‑processing
# ────────────────────────────────────────────────────────────────────────────────
def preprocess_data(
    batch,
    image_processor: SegformerImageProcessor,
    image_col: str = "image",
    mask_col: str = "mask"
):
    """
    Convert raw PIL images & masks into model‑ready tensors.

    • Images are resized/normalized by `SegformerImageProcessor`.
    • Masks are resized using nearest‑neighbour and *kept in palette mode*
      so their pixel values stay in [0, 20] (class indices).
    • Any pixel whose value is not a valid class index is replaced with
      `PASCAL_VOC_IGNORE_INDEX` (255).

    Returned dict keys: "pixel_values" (float tensor Nx3xHxW), "labels" (long tensor NxHxW).
    """
    images = batch[image_col]
    masks  = batch[mask_col]

    # Ensure we always work with lists (HF can pass a single example)
    if isinstance(images, Image.Image):
        images = [images]
    if isinstance(masks, Image.Image):
        masks = [masks]

    # ── 1. Process images ───────────────────────────────────────────────────────
    inputs = image_processor(images, return_tensors="pt")
    pixel_values = inputs["pixel_values"]                    # [N, 3, H, W]

    # ── 2. Process masks ────────────────────────────────────────────────────────
    processed_masks = []
    for mask in masks:
        # If someone saved an RGB mask by mistake, convert to palette (keeps indices)
        if mask.mode == "RGB":
            mask = mask.convert("P")

        # *Do NOT* convert P‑mode to L – that destroys class indices.
        mask_resized = mask.resize(
            (image_processor.size["width"], image_processor.size["height"]),
            Image.NEAREST
        )
        mask_tensor = torch.tensor(np.array(mask_resized), dtype=torch.long)

        # Replace invalid labels with ignore_index
        mask_tensor = torch.where(
            (mask_tensor >= 0) & (mask_tensor < NUM_PASCAL_VOC_LABELS),
            mask_tensor,
            torch.tensor(PASCAL_VOC_IGNORE_INDEX, dtype=torch.long)
        )

        processed_masks.append(mask_tensor)

    labels_tensor = torch.stack(processed_masks, dim=0)      # [N, H, W]

    # Optional safety net: abort early if anything weird slipped through
    invalid = (
        (labels_tensor >= NUM_PASCAL_VOC_LABELS)
        & (labels_tensor != PASCAL_VOC_IGNORE_INDEX)
    )
    if invalid.any():
        raise ValueError(
            f"Found {invalid.sum().item()} out‑of‑range label(s) after preprocessing."
        )

    inputs["labels"] = labels_tensor
    return inputs
