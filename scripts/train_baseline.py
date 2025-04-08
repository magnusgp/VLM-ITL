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
    Trainer,
    SegformerImageProcessor,
    set_seed
)
from datasets import DatasetDict

from utils.config import load_config
from utils.logging import setup_wandb, logger # Use the configured logger
from utils.metrics import compute_metrics_segmentation
from data.pascal_voc import (
    load_pascal_voc_dataset,
    preprocess_data,
    create_train_val_test_splits,
    PASCAL_VOC_ID2LABEL,
    PASCAL_VOC_LABEL2ID,
    NUM_PASCAL_VOC_LABELS
)
from models.segformer import load_model_for_segmentation

def main(config_path: str):
    """Main function for the baseline training script."""
    # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # --- 2. Setup Logging & Seed ---
    run_name = config.get('run_name', 'baseline_run_unnamed')
    setup_wandb(config, run_name=run_name) # Initialize wandb if configured
    set_seed(config['seed'])
    logger.info(f"Seed set to {config['seed']}")
    if torch.cuda.is_available():
        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available. Using CPU.")


    # --- 3. Load and Prepare Data ---
    logger.info("Loading PASCAL VOC dataset...")
    # Load the raw dataset
    raw_datasets = load_pascal_voc_dataset(
        dataset_name=config['dataset']['name'],
        cache_dir=config['dataset'].get('cache_dir') # Optional cache dir
    )

    # Split the original 'train' data into train/val/test for the baseline experiment
    # The original 'validation' split from HF is treated as a hold-out test set by load_pascal_voc_dataset
    # We re-split the 'train' split to get our own val/test sets if specified percentages are > 0
    train_val_test_needed = config['dataset'].get('val_split_percentage', 0) > 0 or \
                             config['dataset'].get('test_split_percentage', 0) > 0

    if train_val_test_needed:
        logger.info("Splitting 'train' dataset into train/validation/test sets...")
        train_ds, val_ds, test_ds = create_train_val_test_splits(
            raw_datasets['train'],
            val_percentage=config['dataset'].get('val_split_percentage', 0.1),
            test_percentage=config['dataset'].get('test_split_percentage', 0.1), # Using a test split from train
            seed=config['seed']
        )
        # Keep the original test set aside if needed, or overwrite with the split one
        # Let's use the split one for consistency in the baseline evaluation
        prepared_datasets = DatasetDict({
            'train': train_ds,
            'validation': val_ds,
            'test': test_ds
        })
        logger.info("Using train/val/test splits derived from the original 'train' split.")
    else:
        # Use original train/test splits if no percentages given (test is original val)
        logger.info("Using original 'train' and 'test' (renamed 'validation') splits.")
        prepared_datasets = DatasetDict({
            'train': raw_datasets['train'],
             # No validation set in this case, Trainer can handle this
            'test': raw_datasets['test']
        })
        # Create a dummy validation set if needed by evaluation strategy
        if config['training']['evaluation_strategy'] != "no":
             logger.warning("No validation split specified. Using test set for evaluation during training.")
             prepared_datasets['validation'] = raw_datasets['test'] # Use test set for eval

    logger.info(f"Final dataset splits: {prepared_datasets}")

    # --- 4. Load Image Processor and Preprocess Data ---
    logger.info("Loading Image Processor...")
    image_processor = SegformerImageProcessor.from_pretrained(
        config['dataset']['feature_extractor_name'],
        do_reduce_labels=False # Handle labels 0-20 correctly
    )

    logger.info("Applying preprocessing to datasets...")
    # Create a partial function for map
    preprocess_fn = partial(
        preprocess_data,
        image_processor=image_processor,
        image_col=config['dataset']['image_col'],
        mask_col=config['dataset']['mask_col']
    )

    # Apply preprocessing using `map`
    processed_datasets = prepared_datasets.map(
        preprocess_fn,
        batched=True,
        batch_size=config['training']['per_device_train_batch_size'] * 2, # Process faster
        remove_columns=raw_datasets['train'].column_names # Remove old columns
    )

    # Set format to PyTorch tensors
    processed_datasets.set_format("torch")
    logger.info("Preprocessing complete.")
    logger.info(f"Processed dataset features: {processed_datasets['train'].features}")


    # --- 5. Load Model ---
    logger.info("Loading segmentation model...")
    model = load_model_for_segmentation(
        model_name_or_path=config['model']['name'],
        num_labels=NUM_PASCAL_VOC_LABELS,
        id2label=PASCAL_VOC_ID2LABEL,
        label2id=PASCAL_VOC_LABEL2ID,
        ignore_mismatched_sizes=config['model'].get('ignore_mismatched_sizes', False)
    )

    # --- 6. Configure Training Arguments ---
    logger.info("Configuring Training Arguments...")
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        run_name=run_name, # Set run name for logging if supported
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
        logging_dir=os.path.join(config['output_dir'], 'logs'), # Log to output dir
        logging_steps=int(config['training']['logging_steps']),
        remove_unused_columns=config['training'].get('remove_unused_columns', False),
        fp16=bool(config['training'].get('fp16', False) and torch.cuda.is_available()),
        seed=config['seed'],
        report_to=config.get('log_with', 'none').split(','), # Support multiple reporters e.g. "wandb,tensorboard"
        push_to_hub=bool(config['training'].get('push_to_hub', False)),
        # Add other TrainingArguments as needed from config
    )
    logger.info(f"FP16 enabled: {training_args.fp16}")


    # --- 7. Initialize Trainer ---
    logger.info("Initializing Trainer...")
    # Create partial function for compute_metrics
    compute_metrics_fn = partial(
        compute_metrics_segmentation,
        num_labels=NUM_PASCAL_VOC_LABELS,
        ignore_index=255 # SegformerImageProcessor might pad with 255
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets.get("validation"), # Optional, uses test if 'validation' not present & eval needed
        compute_metrics=compute_metrics_fn,
        # Data collator is usually handled internally for CV tasks if data is processed correctly
    )

    # --- 8. Train ---
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info("Training finished.")

    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state() # Saves tokenizer, config, model, etc.
    logger.info(f"Training metrics: {metrics}")


    # --- 9. Evaluate ---
    if processed_datasets.get("test") is not None:
        logger.info("Starting evaluation on the test set...")
        eval_metrics = trainer.evaluate(eval_dataset=processed_datasets["test"])

        # Log evaluation metrics
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics) # Saves to output_dir/eval_results.json
        logger.info(f"Evaluation metrics on test set: {eval_metrics}")
    else:
        logger.warning("No test set found in processed datasets. Skipping final evaluation.")


    # --- 10. Save Final Model & Finish Logging ---
    logger.info("Saving the final model...")
    trainer.save_model(os.path.join(config['output_dir'], "final_model")) # Save best model (if load_best_model_at_end=True)
    logger.info(f"Model saved to {os.path.join(config['output_dir'], 'final_model')}")

    # Finish wandb run if it was initialized
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

    # Setup basic logging configuration for the script itself
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