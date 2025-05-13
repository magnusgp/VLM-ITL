import os
import sys
import argparse
import logging
from functools import partial

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import (
    TrainingArguments,
    Trainer,  # Import base Trainer
    SegformerImageProcessor,
    set_seed
)
from datasets import DatasetDict

from utils.config import load_config
from utils.log_utils import setup_wandb, logger
from utils.metrics import compute_metrics_segmentation
from data.pascal_voc import (
    load_pascal_voc_dataset,
    preprocess_data,
    create_train_val_test_splits,
    PASCAL_VOC_LABEL_NAMES,
    PASCAL_VOC_ID2LABEL,
    PASCAL_VOC_LABEL2ID,
    NUM_PASCAL_VOC_LABELS,
    PASCAL_VOC_IGNORE_INDEX  # Make sure ignore index is accessible if needed
)
from models.segformer import load_model_for_segmentation

def debug_log_and_plot(images: torch.Tensor,
                       masks:  torch.Tensor,
                       class_names: list[str],
                       out_path:  str = "debug_segmentation.png") -> None:
    """Helper function to log and plot images and masks for debugging.

    Args:
        images (torch.Tensor): _images_ tensor of shape [B, C, H, W].
        masks (torch.Tensor): _masks_ tensor of shape [B, H, W].
        class_names (list[str]): _class_names_ list of class names.
        out_path (str, optional): _out_path_ path to save the debug image. Defaults to "debug_segmentation.png".
    Returns:
        None
    """
    logger.info("Debugging: Logging and plotting images and masks.")
    if images.dim() != 4 or masks.dim() != 3:
        logger.error("Invalid dimensions for images or masks.")
        raise ValueError("Images must be 4D [B, C, H, W] and masks must be 3D [B, H, W].")
    num_images = images.shape[0]
    if num_images > 4:
        logger.warning("More than 4 images provided. Only the first 4 will be plotted.")
        num_images = 4
    fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 5))
    for i in range(num_images):
        # Plot image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Image {i + 1}")
        # Write label
        label = masks[i].cpu().numpy()
        label = np.unique(label)
        label = [class_names[l] for l in label if l != PASCAL_VOC_IGNORE_INDEX]
        axes[i, 1].imshow(img)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"Label {i + 1}: {', '.join(label)}")
        # Plot mask
        mask = masks[i].cpu().numpy()
        axes[i, 2].imshow(mask, cmap='jet', alpha=0.5)
        axes[i, 2].axis('off')
        axes[i, 2].set_title(f"Mask {i + 1}")
        # Add color legend
        for j, class_name in enumerate(class_names):
            axes[i, 2].add_patch(plt.Rectangle((0, 0), 1, 1, color=plt.cm.jet(j / len(class_names)), label=class_name))
        axes[i, 2].legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize='small')
    # Save figure
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"\tWrote debug figure to {out_path}\n")
    sys.exit(1)  # Exit after saving the figure
    return None

class SegmentationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return a tuple checking if labels were provided or not.
        We override this to explicitly pass labels AND retrieve the loss.
        """
        # Extract labels so we can pass them in the forward pass
        labels = inputs.pop("labels", None)
        pixel_values = inputs.get("pixel_values")

        if labels is None:
            logger.warning("No labels provided in inputs. Loss cannot be computed.")
            raise ValueError("Labels must be provided in inputs to compute loss.")
        if pixel_values is None:
            logger.warning("No pixel values provided in inputs. Loss cannot be computed.")
            raise ValueError("Both 'pixel_values' and 'labels' must be in inputs to compute loss.")
        # Ensure labels are 3D: [batch_size, height, width]
        # The cross_entropy loss expects targets without the channel dimension
        if labels.dim() == 4 and labels.shape[1] == 1:
            logger.info(f"Labels tensor has shape {labels.shape}. Squeezing the channel dimension.")
            labels = labels.squeeze(1)
            logger.info(f"Labels tensor shape after squeeze: {labels.shape}")
        elif labels.dim() != 3:
             # Add a check for other unexpected shapes
             logger.error(f"Unexpected labels tensor shape: {labels.shape}. Expected 3D [B, H, W] or 4D [B, 1, H, W].")
             raise ValueError(f"Unexpected labels tensor shape: {labels.shape}")
        # Forward pass with explicit labels
        outputs = model(pixel_values=pixel_values, labels=labels)

        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            logger.warning("Model output did not contain 'loss' attribute despite labels being provided.")
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                raise ValueError("Could not retrieve loss from model output.")

        return (loss, outputs) if return_outputs else loss


def main(config_path: str):
    """Main function for the baseline training script."""
    # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # --- 2. Setup Logging & Seed ---
    run_name = config.get('run_name', 'baseline_run_unnamed')
    setup_wandb(config, run_name=run_name)
    set_seed(config['seed'])
    logger.info(f"Seed set to {config['seed']}")
    if torch.cuda.is_available():
        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available. Using CPU.")

    # --- 3. Load and Prepare Data ---
    logger.info("Loading PASCAL VOC dataset...")
    raw_datasets = load_pascal_voc_dataset(
        dataset_name=config['dataset']['name'],  # Use name from config
        cache_dir=config['dataset'].get('cache_dir')
    )

    train_val_test_needed = (
        config['dataset'].get('val_split_percentage', 0) > 0
        or config['dataset'].get('test_split_percentage', 0) > 0
    )
    if train_val_test_needed:
        logger.info("Splitting 'train' dataset into train/validation/test sets...")
        train_ds, val_ds, test_ds = create_train_val_test_splits(
            raw_datasets['train'],
            val_percentage=config['dataset'].get('val_split_percentage', 0.1),
            test_percentage=config['dataset'].get('test_split_percentage', 0.1),
            seed=config['seed']
        )
        prepared_datasets = DatasetDict({
            'train': train_ds,
            'validation': val_ds,
            'test': test_ds
        })
        logger.info("Using train/val/test splits derived from the original 'train' split.")
    else:
        logger.info("Using original 'train' and 'test' (renamed 'validation') splits.")
        prepared_datasets = DatasetDict({
            'train': raw_datasets['train'],
            'test': raw_datasets['test']
        })
        if config['training']['evaluation_strategy'] != "no":
            logger.warning("No validation split specified. Using test set for evaluation during training.")
            prepared_datasets['validation'] = raw_datasets['test']

    logger.info(f"Final dataset splits: {prepared_datasets}")

    # --- 4. Load Image Processor and Preprocess Data ---
    logger.info("Loading Image Processor...")
    image_processor = SegformerImageProcessor.from_pretrained(
        config['dataset']['feature_extractor_name'],
        size={"height": 512, "width": 512},
        do_resize=True,
        do_reduce_labels=False,
    )

    logger.info("Applying preprocessing to datasets...")
    preprocess_fn = partial(
        preprocess_data,
        image_processor=image_processor,
        image_col=config['dataset']['image_col'],
        mask_col=config['dataset']['mask_col']
    )
    if config.get('debug', False):

        # Log some sample shapes before mapping
        sample_examples = prepared_datasets["train"].select(range(3))
        for i, example in enumerate(sample_examples):
            processed = preprocess_data(
                example,
                image_processor=image_processor,
                image_col=config['dataset']['image_col'],
                mask_col=config['dataset']['mask_col']
            )
            processed['pixel_values'] = processed['pixel_values'][0]
            processed['labels'] = processed['labels'][0]
            logger.info(f"Sample processed pixel_values shape: {processed['pixel_values'].shape}")
            logger.info(f"Sample processed labels shape: {processed['labels'].shape}")
            uniq = np.unique(processed['labels'])
            # apply id2label to get the label names
            label_names = [PASCAL_VOC_ID2LABEL[i] for i in uniq if i in PASCAL_VOC_ID2LABEL]
            logger.info(f"Sample processed labels unique values: {uniq}")
            logger.info(f"Sample processed labels unique names: {label_names}")
            logger.info("Debugging mode enabled. Saving processed example.")
            logger.info(f"Image {i}:")
            logger.info(f"Processed image shape: {processed['pixel_values'].shape}")
            processed_image = processed['pixel_values'].permute(1, 2, 0).numpy()
            processed_image = (processed_image - processed_image.min()) / (processed_image.max() - processed_image.min())
            plt.imsave(
                os.path.join(config['output_dir'], f"sample_{i}_image.png"),
                processed_image
            )
            # sys.exit(1)

    processed_datasets = prepared_datasets.map(
        preprocess_fn,
        batched=True,
        batch_size=config['training']['per_device_train_batch_size'],
        load_from_cache_file=config['dataset'].get('load_from_cache_file', False),
    )
    processed_datasets.set_format("torch", columns=["pixel_values", "labels"])
    logger.info("Preprocessing complete.")
    logger.info(f"Processed dataset features: {processed_datasets['train'].features}")

    # --- 5. Load Model ---
    logger.info("Loading segmentation model...")
    model = load_model_for_segmentation(
        model_name_or_path=config['model']['name'],
        num_labels=NUM_PASCAL_VOC_LABELS,
        id2label=PASCAL_VOC_ID2LABEL,
        label2id=PASCAL_VOC_LABEL2ID,
        ignore_mismatched_sizes=config['model'].get('ignore_mismatched_sizes', False),
        loss_ignore_index=PASCAL_VOC_IGNORE_INDEX
    )

    # --- 6. Configure Training Arguments ---
    logger.info("Configuring Training Arguments...")
    try:
        training_args = TrainingArguments(
        output_dir=config['output_dir'],
        run_name=run_name,
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=int(config['training']['per_device_train_batch_size']),
        per_device_eval_batch_size=int(config['training']['per_device_eval_batch_size']),
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=int(config['training']['save_total_limit']),
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        logging_dir=os.path.join(config['output_dir'], 'logs'),
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        lr_scheduler_kwargs=config['training'].get('lr_scheduler_kwargs', {}),
        warmup_ratio=float(config['training'].get('warmup_ratio', 0.0)),
        # early_stopping_patience=int(config['training'].get('early_stopping_patience', 0)),
        logging_steps=int(config['training']['logging_steps']),
        remove_unused_columns=config['training'].get('remove_unused_columns', False),
        fp16=bool(config['training'].get('fp16', False) and torch.cuda.is_available()),
        seed=config['seed'],
        report_to=config.get('log_with', 'none').split(','),  # e.g. "wandb,tensorboard"
        push_to_hub=bool(config['training'].get('push_to_hub', False)),
    )
    except TypeError as e:
        logging.error(f"Transformers module mismatch. Please check the version. ({e})")    
        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            run_name=run_name,
            num_train_epochs=config['training']['num_train_epochs'],
            per_device_train_batch_size=int(config['training']['per_device_train_batch_size']),
            per_device_eval_batch_size=int(config['training']['per_device_eval_batch_size']),
            learning_rate=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay']),
            evaluation_strategy=config['training']['evaluation_strategy'],
            save_strategy=config['training']['save_strategy'],
            save_total_limit=int(config['training']['save_total_limit']),
            load_best_model_at_end=config['training']['load_best_model_at_end'],
            metric_for_best_model=config['training']['metric_for_best_model'],
            logging_dir=os.path.join(config['output_dir'], 'logs'),
            lr_scheduler_type=config['training']['lr_scheduler_type'],
            logging_steps=int(config['training']['logging_steps']),
            remove_unused_columns=config['training'].get('remove_unused_columns', False),
            fp16=bool(config['training'].get('fp16', False) and torch.cuda.is_available()),
            seed=config['seed'],
            report_to=config.get('log_with', 'none').split(','),  # e.g. "wandb,tensorboard"
            push_to_hub=bool(config['training'].get('push_to_hub', False)),
        )
    logger.info(f"FP16 enabled: {training_args.fp16}")
    logger.info(f"Using remove_unused_columns: {training_args.remove_unused_columns}")

    # --- 7. Initialize Trainer ---
    logger.info("Initializing Trainer...")
    compute_metrics_fn = partial(
        compute_metrics_segmentation,
        num_labels=NUM_PASCAL_VOC_LABELS,
        ignore_index=PASCAL_VOC_IGNORE_INDEX
    )

    trainer = SegmentationTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets.get("validation"),
        compute_metrics=compute_metrics_fn
    )
    
    if config.get('debug', False):
        logger.info("Debugging mode enabled. Logging a batch of data.")
        # Log a batch of data
        train_loader = trainer.get_train_dataloader()
        batch = next(iter(train_loader))
        imgs = batch["pixel_values"]   # tensor [B,3,H,W]
        msks = batch["labels"]         # tensor [B,H,W]
        debug_log_and_plot(imgs, msks, PASCAL_VOC_LABEL_NAMES)

    # --- 8. Train ---
    logger.info("Starting training...")
    train_result = trainer.train()  # Uses custom compute_loss
    logger.info("Training finished.")
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info(f"Training metrics: {metrics}")

    # Final evaluation on test set if present
    if processed_datasets.get("test") is not None:
        logger.info("Starting evaluation on the test set...")
        eval_metrics = trainer.evaluate(eval_dataset=processed_datasets["test"])
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        logger.info(f"Evaluation metrics on test set: {eval_metrics}")
    else:
        logger.warning("No test set found. Skipping final evaluation.")

    logger.info("Saving the final model...")
    trainer.save_model(os.path.join(config['output_dir'], "final_model"))
    logger.info(f"Model saved to {os.path.join(config['output_dir'], 'final_model')}")

    if config.get('log_with') == 'wandb' and trainer.is_world_process_zero():
        import wandb
        wandb.finish()

    logger.info("Baseline training script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline Image Segmentation Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the baseline configuration YAML file."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    try:
        main(args.config)
    except Exception as e:
        logger.error("An error occurred during the baseline training pipeline.", exc_info=True)
        sys.exit(1)
