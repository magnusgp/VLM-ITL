from transformers import SegformerForSemanticSegmentation, SegformerConfig
from typing import Dict
import logging
import torch

from data.pascal_voc import PASCAL_VOC_ID2LABEL, PASCAL_VOC_LABEL2ID, PASCAL_VOC_IGNORE_INDEX # Import mappings

logger = logging.getLogger(__name__)

def load_model_for_segmentation(
    model_name_or_path: str,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    ignore_mismatched_sizes: bool = False,
    loss_ignore_index: int = PASCAL_VOC_IGNORE_INDEX
) -> SegformerForSemanticSegmentation:
    """
    Loads a SegFormer model for semantic segmentation, configured for the specific dataset.

    Args:
        model_name_or_path (str): The name of the pre-trained model or path to local checkpoint.
        num_labels (int): The number of output segmentation classes.
        id2label (Dict[int, str]): Mapping from label ID to label name.
        label2id (Dict[str, int]): Mapping from label name to label ID.
        ignore_mismatched_sizes (bool): Whether to ignore size mismatches when loading
                                        pre-trained weights (useful if changing classifier head).
        loss_ignore_index (int): Label index to be ignored by the loss function.

    Returns:
        SegformerForSemanticSegmentation: The loaded and configured model.

    Raises:
        ValueError: If the number of labels doesn't match the provided mappings.
        RuntimeError: If the model fails to load.
    """
    if num_labels != len(id2label) or num_labels != len(label2id):
        raise ValueError(f"num_labels ({num_labels}) must match the size of id2label ({len(id2label)}) and label2id ({len(label2id)}).")

    logger.info(f"Loading SegFormer model: '{model_name_or_path}' for {num_labels} classes.")
    logger.info(f"Using id2label mapping: {id2label}")
    logger.info(f"Setting loss ignore index to: {loss_ignore_index}") # Log the ignore index

    try:
        # Load the model configuration and update it for the specific dataset
        config = SegformerConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            loss_ignore_index=loss_ignore_index
        )

        # Load the pre-trained model with the updated config
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name_or_path,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes, # Crucial for fine-tuning on a different number of classes
        )
        logger.info("SegFormer model loaded successfully.")
        # Log model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")

        return model

    except Exception as e:
        logger.error(f"Failed to load SegFormer model '{model_name_or_path}': {e}", exc_info=True)
        raise RuntimeError(f"Model loading failed for {model_name_or_path}") from e


if __name__ == "__main__":
    print("Testing SegFormer model loading...")

    # Use PASCAL VOC settings
    num_classes = NUM_PASCAL_VOC_LABELS
    id2lbl = PASCAL_VOC_ID2LABEL
    lbl2id = PASCAL_VOC_LABEL2ID
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512" # Example pre-trained model

    try:
        model = load_model_for_segmentation(
            model_name_or_path=model_name,
            num_labels=num_classes,
            id2label=id2lbl,
            label2id=lbl2id,
            ignore_mismatched_sizes=True # Important because we change head from ADE20k (150) to PASCAL (21)
        )
        print("\nModel loaded successfully:")
        # print(model) # Can be very verbose

        # Verify config matches
        assert model.config.num_labels == num_classes
        assert model.config.id2label == id2lbl
        assert model.config.label2id == lbl2id
        assert model.model.decoder.classifier.out_channels == num_classes # Check classifier output channels

        print("\nModel configuration verified.")

        # Test with a dummy input (requires torch)
        try:
            import torch
            # Dummy input: Batch=1, Channels=3, Height=512, Width=512 (common SegFormer size)
            dummy_input = torch.randn(1, 3, 512, 512)
            with torch.no_grad():
                outputs = model(pixel_values=dummy_input)
            logits = outputs.logits
            # Logits shape: [Batch, NumClasses, H/4, W/4] for SegFormer
            print(f"\nModel forward pass successful. Logits shape: {logits.shape}")
            assert list(logits.shape) == [1, num_classes, 512 // 4, 512 // 4]
            print("Output shape verified.")

        except ImportError:
            print("\nSkipping forward pass test: PyTorch not fully available.")
        except Exception as e:
            print(f"\nError during model forward pass test: {e}")

    except Exception as e:
        print(f"\nError during model loading test: {e}")