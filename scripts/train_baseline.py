import os
import sys
import argparse
import logging
from functools import partial

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
from transformers import (
    TrainingArguments,
    Trainer,  # Import base Trainer
    SegformerImageProcessor,
    set_seed
)
from datasets import DatasetDict

from utils.config import load_config
from utils.logging import setup_wandb, logger
from utils.metrics import compute_metrics_segmentation
from data.pascal_voc import (
    load_pascal_voc_dataset,
    preprocess_data,
    create_train_val_test_splits,
    PASCAL_VOC_ID2LABEL,
    PASCAL_VOC_LABEL2ID,
    NUM_PASCAL_VOC_LABELS,
    PASCAL_VOC_IGNORE_INDEX  # Make sure ignore index is accessible if needed
)
from models.segformer import load_model_for_segmentation


class SegmentationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return a tuple checking if labels were provided or not.
        We override this to explicitly pass labels AND retrieve the loss.
        """
        # logger.info("Inputs to compute_loss: %s", inputs)
        # logger.info(f"Inputs keys: {inputs.keys()}")

        # Extract labels so we can pass them in the forward pass
        labels = inputs.pop("labels", None)
        pixel_values = inputs.get("pixel_values")

        if labels is None:
            logger.warning("No labels provided in inputs. Loss cannot be computed.")
            raise ValueError("Labels must be provided in inputs to compute loss.")
        if pixel_values is None:
            logger.warning("No pixel values provided in inputs. Loss cannot be computed.")
            raise ValueError("Both 'pixel_values' and 'labels' must be in inputs to compute loss.")
        # --- START FIX ---
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
        # --- END FIX ---
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
        do_reduce_labels=True
    )

    logger.info("Applying preprocessing to datasets...")
    preprocess_fn = partial(
        preprocess_data,
        image_processor=image_processor,
        image_col=config['dataset']['image_col'],
        mask_col=config['dataset']['mask_col']
    )

    # Log some sample shapes before mapping
    sample_examples = prepared_datasets["train"].select(range(2))
    for example in sample_examples:
        processed = preprocess_data(
            example,
            image_processor=image_processor,
            image_col=config['dataset']['image_col'],
            mask_col=config['dataset']['mask_col']
        )
        logger.info(f"Sample processed pixel_values shape: {processed['pixel_values'].shape}")
        logger.info(f"Sample processed labels shape: {processed['labels'].shape}")

    processed_datasets = prepared_datasets.map(
        preprocess_fn,
        batched=True,
        batch_size=config['training']['per_device_train_batch_size'],
        load_from_cache_file=True
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
