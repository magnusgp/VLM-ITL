# from dotenv import load_dotenv
# load_dotenv()
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


PASCAL_VOC_BINARY_LABEL_NAMES = [
    "background", "foreground"
]
PASCAL_VOC_BINARY_ID2LABEL = {i: label for i, label in enumerate(PASCAL_VOC_BINARY_LABEL_NAMES)}
PASCAL_VOC_BINARY_LABEL2ID = {label: i for i, label in PASCAL_VOC_BINARY_ID2LABEL.items()}
NUM_PASCAL_VOC_BINARY_LABELS = len(PASCAL_VOC_BINARY_ID2LABEL)


# Pixels with this value will be ignored by Cross‑Entropy
PASCAL_VOC_IGNORE_INDEX = 255

from PIL import Image

PASCAL_VOC_BINARY_COLORS = [
    (0, 0, 0),   # 0=background
    (255, 0, 0)  # 1=foreground
]


PASCAL_VOC_COLORS = [
    (  0,   0,   0),  #  0=background
    (128,   0,   0),  #  1=aeroplane
    (  0, 128,   0),  #  2=bicycle
    (128, 128,   0),  #  3=bird
    (  0,   0, 128),  #  4=boat
    (128,   0, 128),  #  5=bottle
    (  0, 128, 128),  #  6=bus
    (128, 128, 128),  #  7=car
    ( 64,   0,   0),  #  8=cat
    (192,   0,   0),  #  9=chair
    ( 64, 128,   0),  # 10=cow
    (192, 128,   0),  # 11=diningtable
    ( 64,   0, 128),  # 12=dog
    (192,   0, 128),  # 13=horse
    ( 64, 128, 128),  # 14=motorbike
    (192, 128, 128),  # 15=person
    (  0,  64,   0),  # 16=potted plant
    (128,  64,   0),  # 17=sheep
    (  0, 192,   0),  # 18=sofa
    (128, 192,   0),  # 19=train
    (  0,  64, 128)   # 20=tv/monitor
]

# ────────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ────────────────────────────────────────────────────────────────────────────────
def load_pascal_voc_dataset(
    dataset_name: str = "nateraw/pascal-voc-2012",
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Load the Pascal-VOC 2012 dataset from HuggingFace Datasets.

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
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a single HF Dataset into train/val/test subsets.
    """
    num_samples = len(full_dataset)

    indices = list(range(num_samples))
    random.seed(seed)
    random.shuffle(indices)

    num_val  = int(num_samples * val_percentage)

    val_indices  = indices[:num_val]
    train_indices = indices[num_val:]

    logger.info(
        f"Split dataset → Train={len(train_indices)}, "
        f"Val={len(val_indices)}"
    )
    return train_indices, val_indices


# ────────────────────────────────────────────────────────────────────────────────
# Pre‑processing
# ────────────────────────────────────────────────────────────────────────────────

def preprocess_data(
    batch: Dict[str, Any],
    image_processor: SegformerImageProcessor,
    image_col: str = "image",
    mask_col: str = "mask",
    binary_segmentation_task: bool = False  # New parameter
) -> Dict[str, List[Any]]:
    """
    Preprocess a batch of Pascal VOC examples for segmentation.

    - Applies the Huggingface image_processor to resize/normalize inputs.
    - Converts the ground-truth RGB segmentation masks into
      class-index masks using the official VOC 21-color palette.
    - Marks any pixel whose color is not in the VOC palette as IGNORE_INDEX.
    - If binary_segmentation_task is True, maps all foreground classes to 1.
    - Returns lists of pixel_values and label masks (len == batch_size).

    Args:
        batch (Dict[str, Any]):
            A batch from HF datasets containing:
              - "image": List[PIL.Image] RGB inputs
              - "mask": List[PIL.Image] RGB segmentation masks
        image_processor (SegformerImageProcessor):
            Huggingface processor for resizing/normalizing images.
        image_col (str): Name of the image column.
        mask_col (str): Name of the mask column.
        binary_segmentation_task (bool): If True, convert to binary segmentation.

    Returns:
        Dict[str, List[Any]]:
            {
              "pixel_values": List[np.ndarray] of shape (C,H,W),
              "labels":       List[np.ndarray] of shape (H,W) with dtype np.int64
            }
    """
    images = batch[image_col]
    masks = batch[mask_col]

    if not isinstance(images, list):
        images = [images]
    if not isinstance(masks, list):
        masks = [masks]
    
    # Sanity check: ensure images and masks are the same length
    if len(images) != len(masks):
        raise ValueError(
            f"Batch length mismatch: {len(images)} images vs {len(masks)} masks."
        )

    # Process images in a batch (returns NumPy by default)
    processed = image_processor(
        images,
        return_tensors="pt"
    )
    pixel_values_batch = processed["pixel_values"]  # shape: (B,C,H,W)

    label_masks: List[np.ndarray] = []

    # target size after processing
    _, _, H, W = pixel_values_batch.shape

    for idx, mask in enumerate(masks):
        # Ensure PIL image
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask, dtype=np.uint8))

        # Resize mask to match model input (nearest for segmentation)
        mask_resized = mask.convert("RGB").resize((W, H), Image.NEAREST)
        arr = np.array(mask_resized, dtype=np.uint8)  # shape (H, W, 3)

        # Build class-index map, default to IGNORE_INDEX
        idx_map = np.full((H, W), PASCAL_VOC_IGNORE_INDEX, dtype=np.int64)

        # Map each VOC color → class index
        for class_id, color in enumerate(PASCAL_VOC_COLORS):
            matches = np.all(arr == color, axis=-1)
            if np.any(matches):
                idx_map[matches] = class_id

        # If binary segmentation, map all foreground classes to 1
        if binary_segmentation_task:
            # Foreground pixels are those not background (0) and not ignore_index
            is_foreground = (idx_map != 0) & (idx_map != PASCAL_VOC_IGNORE_INDEX)
            idx_map[is_foreground] = 1 # Map to class 1 (foreground)

        # Sanity check: ensure only valid IDs or IGNORE_INDEX appear
        unique_vals = np.unique(idx_map)
        bad = [int(v) for v in unique_vals
               if v != PASCAL_VOC_IGNORE_INDEX and (v < 0 or v >= NUM_PASCAL_VOC_LABELS)]
        if bad:
            raise ValueError(
                f"Invalid class IDs {bad} in mask #{idx} "
                f"(allowed 0..{NUM_PASCAL_VOC_LABELS-1} or {PASCAL_VOC_IGNORE_INDEX})."
            )

        label_masks.append(idx_map)

    # Convert pixel_values_batch to list to satisfy arrow‐map requirements
    pixel_list = [pixel_values_batch[i].contiguous() for i in range(pixel_values_batch.shape[0])]
    label_masks = torch.tensor(np.array(label_masks, dtype=np.int64), dtype=torch.int64)

    return {
        "pixel_values": pixel_list,
        "labels":       label_masks
    }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import cv2
    from tqdm import tqdm
    from PIL import ImageDraw
    # Log a few dataset examples and save the image along with the labels in the image masks
    dataset = load_pascal_voc_dataset()
    dataset = dataset["train"].shuffle(seed=42).select(range(5))
    image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    dataset = dataset.map(
        lambda batch: preprocess_data(batch, image_processor),
        batched=True,
        remove_columns=["image", "mask"],
        desc="Preprocessing images & masks"
    )
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])
    # plot the dataset images and save the figure
    
    for i in tqdm(range(len(dataset))):
        image = dataset[i]["pixel_values"].permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        mask = dataset[i]["labels"].numpy()
        mask = np.where(mask == PASCAL_VOC_IGNORE_INDEX, 0, mask)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
        plt.imshow(overlay)
        plt.axis("off")
        plt.savefig(f"overlay_{i}.png", bbox_inches="tight")
        plt.close()
        
    # Save the images and masks
    os.makedirs("images", exist_ok=True)
    os.makedirs("masks", exist_ok=True)
    os.makedirs("overlays", exist_ok=True)
    os.makedirs("images_with_masks", exist_ok=True)
    
    for i in tqdm(range(len(dataset))):
        image = dataset[i]["pixel_values"].permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        mask = dataset[i]["labels"].numpy()
        mask = np.where(mask == PASCAL_VOC_IGNORE_INDEX, 0, mask)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Save the images and masks
        cv2.imwrite(f"images/image_{i}.png", image)
        cv2.imwrite(f"masks/mask_{i}.png", mask)
        # Save the overlay
        overlay = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
        cv2.imwrite(f"overlays/overlay_{i}.png", overlay)
        # Save the image with the mask
        pil_image = Image.fromarray(image)
        pil_mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(pil_image)
        draw.bitmap((0, 0), pil_mask, fill=(255, 0, 0))
        pil_image.save(f"images_with_masks/image_{i}.png")
        
    # close and exit the program
    plt.close("all")
    cv2.destroyAllWindows()
    exit(0)
