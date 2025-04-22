from datasets import load_dataset, Dataset, DatasetDict
from transformers import SegformerImageProcessor
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from PIL import Image
import random
import logging
import torch
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Define PASCAL VOC labels (adjust if needed based on specific dataset version)
# Usually 0: background, 1: aeroplane, ..., 20: tvmonitor
PASCAL_VOC_LABEL_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
PASCAL_VOC_ID2LABEL = {i: label for i, label in enumerate(PASCAL_VOC_LABEL_NAMES)}
PASCAL_VOC_LABEL2ID = {label: i for i, label in PASCAL_VOC_ID2LABEL.items()}
NUM_PASCAL_VOC_LABELS = len(PASCAL_VOC_ID2LABEL)

PASCAL_VOC_IGNORE_INDEX = 255 # Define the ignore index

def load_pascal_voc_dataset(
    dataset_name: str = "nateraw/pascal-voc-2012",
    cache_dir: Optional[str] = None
) -> DatasetDict:
    """
    Loads the PASCAL VOC dataset from Hugging Face datasets.

    Args:
        dataset_name (str): The name of the dataset on the Hugging Face Hub.
        cache_dir (Optional[str]): Directory to cache the downloaded dataset.

    Returns:
        DatasetDict: A dictionary containing the 'train', 'validation', and 'test' splits.
                     Note: 'merve/pascal-voc' only has 'train' and 'validation'. We'll split
                     'train' further if needed for baseline/AL validation/test.
    """
    logger.info(f"Loading dataset '{dataset_name}'...")
    try:
        # Load the dataset, splits might need adjustment
        # 'merve/pascal-voc' has 'train' and 'validation'. Let's treat 'validation' as the test set
        # and split 'train' into train/val for our purposes.
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)

        if "train" not in dataset or "validation" not in dataset:
            if "val" not in dataset:
                raise ValueError(f"Dataset '{dataset_name}' does not contain 'train' or 'validation' splits.")
            # If 'val' is present, we can rename it to 'validation'
            dataset["validation"] = dataset.pop("val")

        # Rename 'validation' to 'test' for clarity in our pipeline
        dataset["test"] = dataset.pop("validation")

        logger.info("Dataset loaded successfully.")
        logger.info(f"Original splits: {list(dataset.keys())}")
        logger.info(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset '{dataset_name}': {e}", exc_info=True)
        raise


def create_train_val_test_splits(
    full_dataset: Dataset,
    val_percentage: float = 0.1,
    test_percentage: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits a given dataset (assumed to be the 'train' split from load_pascal_voc_dataset)
    into training, validation, and test sets.

    Args:
        full_dataset (Dataset): The dataset to split (e.g., the original 'train' split).
        val_percentage (float): Percentage of data to use for validation (from the original set).
        test_percentage (float): Percentage of data to use for testing (from the original set).
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: train_dataset, val_dataset, test_dataset
    """
    num_samples = len(full_dataset)
    if val_percentage + test_percentage >= 1.0:
        raise ValueError("Sum of val_percentage and test_percentage must be less than 1.0")

    indices = list(range(num_samples))
    random.seed(seed)
    random.shuffle(indices) # Shuffle indices in place

    num_test = int(num_samples * test_percentage)
    num_val = int(num_samples * val_percentage)
    num_train = num_samples - num_val - num_test

    test_indices = indices[:num_test]
    val_indices = indices[num_test : num_test + num_val]
    train_indices = indices[num_test + num_val :]

    train_dataset = full_dataset.select(train_indices)
    val_dataset = full_dataset.select(val_indices)
    test_dataset = full_dataset.select(test_indices)

    logger.info(f"Split dataset: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset


def preprocess_data(
    batch: Dict[str, Any],
    image_processor: SegformerImageProcessor,
    image_col: str = "image",
    mask_col: str = "mask",
    ignore_index: int = PASCAL_VOC_IGNORE_INDEX
) -> Dict[str, Any]:
    """
    Preprocesses a batch of data using the SegformerImageProcessor.

    Args:
        batch (Dict[str, Any]): A dictionary representing a batch of data,
                                 containing keys for image and mask columns.
        image_processor (SegformerImageProcessor): The processor to use for transforming data.
        image_col (str): The name of the column containing PIL Image objects.
        mask_col (str): The name of the column containing PIL Image segmentation masks.

    Returns:
        Dict[str, Any]: The processed batch, typically containing 'pixel_values' and 'labels'.
    """
    logger.debug(f"Columns: {batch.keys()}, Image Col: {image_col}, Mask Col: {mask_col}")
    # Extract images and masks from the batch
    images = batch[image_col]
    segmentation_masks = batch[mask_col]

    try: # Process images to get pixel_values
        if not isinstance(images, (list, tuple)):
            images = [images]

        inputs = image_processor(
            images,
            return_tensors="pt"
        )
        # For masks, since the processor might not process them, add a manual step:
        raw_masks = segmentation_masks
        if not isinstance(raw_masks, (list, tuple)):
            raw_masks = [raw_masks]

        # Log the original mask sizes for debugging
        mask_sizes = [f"{mask.size}" if hasattr(mask, "size") else "N/A" for mask in raw_masks]
        
        # Process masks: convert to grayscale if needed, resize using nearest-neighbor,
        # and then convert to tensor.
        processed_masks = []
        for mask in raw_masks:
            if mask.mode != 'L':
                mask = mask.convert('L')
            mask_resized = mask.resize((image_processor.size["width"], image_processor.size["height"]), Image.NEAREST)
            processed_masks.append(torch.tensor(np.array(mask_resized), dtype=torch.long))
            
        # Stack masks and add a channel dimension
        labels_tensor = torch.stack(processed_masks)
        labels_tensor = labels_tensor.unsqueeze(1)
        inputs["labels"] = labels_tensor

        return inputs
    
    except ValueError as ve:
        logger.error(f"ValueError during image processing: {ve}", exc_info=True)
        try:
            img = images[0]
            msk = processed_masks[0]
            logger.error(f"Failed processing image size {img.size} mode {img.mode}, mask size {msk.size} mode {msk.mode}")
        except:
            pass
        raise # Re-raise the exception
    except Exception as e:
        logger.error(f"Error during image processing: {e}", exc_info=True)
        try:
            img = images[0]
            msk = processed_masks[0]
            logger.error(f"Failed processing image size {img.size} mode {img.mode}, mask size {msk.size} mode {msk.mode}")
        except:
            pass # Ignore if logging the image fails too
        raise # Re-raise the exception


if __name__ == "__main__":
    print("Testing data loading and preprocessing...")

    # 1. Load dataset (might download)
    try:
        pascal_dataset_dict = load_pascal_voc_dataset()
        print("Dataset loaded:")
        print(pascal_dataset_dict)
        # Access a sample
        sample = pascal_dataset_dict["train"][0]
        print("\nSample data point keys:", sample.keys())
        print(f"Image type: {type(sample['image'])}, size: {sample['image'].size}, mode: {sample['image'].mode}")
        print(f"Mask type: {type(sample['mask'])}, size: {sample['mask'].size}, mode: {sample['mask'].mode}")
        # Check mask values (should be within 0-20 or 255 for borders if present)
        mask_arr = np.array(sample['mask'])
        print(f"Mask unique values: {np.unique(mask_arr)}")

    except Exception as e:
        print(f"\nError loading dataset: {e}. Skipping further tests.")
        exit()

    # 2. Test Preprocessing
    try:
        feature_extractor_name = "nvidia/segformer-b0-finetuned-ade-512-512" # Example
        image_processor = SegformerImageProcessor.from_pretrained(feature_extractor_name, do_reduce_labels=False) # PASCAL has 0-20 labels
        print(f"\nLoaded Image Processor: {feature_extractor_name}")

        # Create a small batch for testing
        batch_size = 4
        test_batch = pascal_dataset_dict["train"][:batch_size] # This returns a Dict[str, List]
        print(f"\nKeys in batch: {test_batch.keys()}\n")

        # Preprocess the batch
        processed_batch = preprocess_data(test_batch, image_processor)
        print("\nProcessed batch keys:", processed_batch.keys())
        print("Pixel values shape:", processed_batch['pixel_values'].shape) # Should be [batch, 3, H, W]
        print("Labels shape:", processed_batch['labels'].shape) # Should be [batch, H, W]
        print("Labels data type:", processed_batch['labels'].dtype) # Should be torch.int64 (Long)
        print("Labels unique values:", torch.unique(processed_batch['labels'])) # Check processed label range

        # Test with map function (more realistic usage)
        processed_dataset = pascal_dataset_dict["train"].map(
            preprocess_data,
            batched=True,
            batch_size=2, # Smaller batch size for map testing
            fn_kwargs={"image_processor": image_processor}
        )
        print("\nDataset after .map() preprocessing:")
        print(processed_dataset)
        print("Features:", processed_dataset.features)
        # Check a sample from the processed dataset
        processed_sample = processed_dataset[0]
        print("\nSample from processed dataset:")
        print("Pixel values shape:", type(processed_sample['pixel_values'])) # Should be [3, H, W] - PyTorch tensor now
        print("Labels shape:", type(processed_sample['labels'])) # Should be [H, W] - PyTorch tensor now

        print("\nPreprocessing tests successful.")

    except Exception as e:
        print(f"\nError during preprocessing test: {e}")

    # 3. Test Train/Val/Test Splitting (using the original 'train' split)
    try:
        print("\nTesting train/val/test splitting...")
        train_ds, val_ds, test_ds = create_train_val_test_splits(
            pascal_dataset_dict['train'], # Split the original training data
            val_percentage=0.15,
            test_percentage=0.15,
            seed=123
        )
        print(f"Split sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
        total_split_size = len(train_ds) + len(val_ds) + len(test_ds)
        assert total_split_size == len(pascal_dataset_dict['train'])
        print("Splitting test successful.")
    except Exception as e:
        print(f"\nError during splitting test: {e}")