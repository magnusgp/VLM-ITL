import os
import sys
import argparse
import logging
import math
import random
from functools import partial
from typing import Dict, Any, List
import copy # To deep copy config for modifications per iteration
import torch
import numpy as np
from transformers import (
    TrainingArguments,
    Trainer,
    SegformerImageProcessor,
    set_seed,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict, concatenate_datasets

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from utils.config import load_config
from utils.log_utils import (
    setup_wandb, 
    logger, 
    log_active_learning_summary, 
    debug_log_and_plot
)
from utils.metrics import compute_metrics_segmentation
from utils.active_learning import (
    sample_initial_data, 
    select_next_batch_indices,
    feature_extractor_fn,
    ActiveLearningProgressCallback, 
    SegmentationImageLoggerCallback,
    compute_image_uncertainties,
    compute_mean_iou
)
from data.pascal_voc import (
    load_pascal_voc_dataset,
    preprocess_data,
    create_train_val_test_splits, # Use this for consistent val/test sets
    PASCAL_VOC_LABEL_NAMES,
    PASCAL_VOC_ID2LABEL,
    PASCAL_VOC_LABEL2ID,
    NUM_PASCAL_VOC_LABELS
)
from models.segformer import load_model_for_segmentation

def run_active_learning_pipeline(config_path: str):
    """Main function for the active learning simulation script."""
        # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    al_config = config['active_learning']
    run_name_prefix = config.get('run_name_prefix', 'al_run')
    output_dir_prefix = config.get('output_dir_prefix', './results/active_learning')

    # --- 2. Setup Seed ---
    # Set seed early, affects data splitting and initial sampling
    set_seed(config['seed'])
    logger.info(f"Global seed set to {config['seed']}")

    # --- 3. Load and Prepare Full Data ---
    # Load the raw dataset ONCE
    logger.info("Loading PASCAL VOC dataset...")
    raw_dataset = load_pascal_voc_dataset(
        dataset_name=config['dataset']['name'],
        cache_dir=config['dataset'].get('cache_dir')
    )
    
    logger.info("Creating fixed validation and test sets...")
    train_indices, val_indices, test_indices = create_train_val_test_splits(
        raw_dataset['train'],
        # raw_dataset, # here, load_dataset returns a DatasetDict instead of a Dataset
        val_percentage=config['dataset'].get('val_split_percentage', 0.1),
        test_percentage=config['dataset'].get('test_split_percentage', 0.1),
        seed=config['seed'] # Use the global seed for splitting
    )

    # --- 4. Prepare Image Processor and Preprocessing Function ---
    # Load image processor ONCE
    logger.info("Loading Image Processor...")
    image_processor = SegformerImageProcessor.from_pretrained(
        config['dataset']['feature_extractor_name'],
        do_reduce_labels=False
    )
    # Create partial preprocessing function ONCE
    preprocess_fn = partial(
        preprocess_data,
        image_processor=image_processor,
        image_col=config['dataset']['image_col'],
        mask_col=config['dataset']['mask_col']
    )
    
    # Preprocess the full dataset
    logger.info("Preprocessing full dataset...")
    full_dataset = raw_dataset['train'].map( # to combat DatasetDict issue
        preprocess_fn,
        batched=True,
        batch_size=config['dataset'].get('batch_size', 16),
        remove_columns=[config['dataset']['image_col'], config['dataset']['mask_col']],
        load_from_cache_file=True,
    )
    full_dataset.set_format("torch", columns=["pixel_values", "labels"])
    logger.info(f"Full dataset size: {len(full_dataset)}")
    
    # Split the dataset into train, validation, and test sets
    full_dataset_dict = DatasetDict({
        'train': full_dataset.select(train_indices),
        'validation': full_dataset.select(val_indices),
        'test': full_dataset.select(test_indices)
    })
    
    # --- 5. Initialize Active Learning Loop Variables ---
    overall_metrics = {} # Store metrics per iteration {data_percentage: metrics_dict}
    num_total_train = len(full_dataset_dict['train'])
    logger.info(f"Total training samples: {num_total_train}")
    all_train_indices = list(range(num_total_train))
    random.seed(config['seed']) # Re-seed before shuffling for sampling consistency if needed
    random.shuffle(all_train_indices) # Shuffle indices for random sampling pool

    current_percentage = al_config['initial_percentage']
    increment = al_config['increment_percentage']
    max_percentage = al_config['max_percentage']

    num_initial = math.ceil(num_total_train * current_percentage)
    current_indices = all_train_indices[:num_initial]
    remaining_indices = all_train_indices[num_initial:]

    # Determine number of AL steps
    al_steps = []
    p = current_percentage
    while p <= max_percentage + 1e-6: # Add tolerance for float comparison
        al_steps.append(int(round(p * 100))) # Store percentage integer (e.g., 10, 20)
        p += increment
    total_al_steps = len(al_steps)
    logger.info(f"Planned Active Learning percentages: {al_steps}%")

    # --- 6. Active Learning Loop ---
    current_model = None # Variable to hold the model between iterations
    for i, target_percentage_int in enumerate(al_steps):
        current_al_step = i + 1
        current_data_percentage = target_percentage_int / 100.0

        # --- 6a. Prepare Data for Current Iteration ---
        num_target_samples = math.ceil(num_total_train * current_data_percentage)
        num_current_samples = len(current_indices)
        num_to_add = max(0, num_target_samples - num_current_samples)

        logger.info(f"\n--- Active Learning Iteration {current_al_step}/{total_al_steps} ---")
        logger.info(f"Target Data: {target_percentage_int}% ({num_target_samples} samples)")
        logger.info(f"Current Data: {num_current_samples} samples")

        # --- 6a. Active Learning Sampling (entropy-based) ---
        if num_to_add > 0 and remaining_indices:
            k = min(num_to_add, len(remaining_indices))
            logger.info(f"Selecting {k} new samples via entropy sampling…")

            # Only score the *remaining* (unlabeled) pool
            uncertainties, segmentations = compute_image_uncertainties(
                model=current_model,
                dataset=full_dataset_dict['train'],
                remaining_indices=remaining_indices,
                preprocess_fn=preprocess_fn,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=al_config.get("inference_batch_size", 8),
            )

            # Pick top-k by descending entropy
            new_indices = sorted(
                uncertainties, 
                key=lambda idx: uncertainties[idx], 
                reverse=False if al_config.get("min_entropy", False) else True,
                # sort by ascending entropy instead for min_entropy
                # reverse=True, 
            )[:k]

            # 1) Threshold the IoU
            iou_thresh = al_config.get("iou_threshold", 0.8)
            good_indices = []
            bad_indices  = []

            for idx in new_indices:
                true_mask = full_dataset_dict["train"][idx]["labels"]           # numpy array of shape [H,W]
                pred_mask = segmentations[idx]                          # your model’s numpy [H,W]
                miou = compute_mean_iou(true_mask, pred_mask)
                logger.info(f"  idx={idx}  mean IoU = {miou:.3f}")
                if miou >= iou_thresh:
                    good_indices.append(idx)
                else:
                    bad_indices.append(idx)

            logger.info(f"{len(good_indices)}/{len(new_indices)} passed IoU ≥ {iou_thresh}")

            # 2)  Override the true masks *in place* for just those good examples:
            def _override_mask(example, example_idx):
                if example_idx in good_indices:
                    # turn the H×W numpy array into a list of lists
                    return {"labels": segmentations[example_idx].tolist()}
                else:
                    # return no change
                    return {}

            # this will go over your entire train split, but only rewrite the
            # 'mask' for good_indices
            full_dataset_dict["train"] = full_dataset_dict["train"].map(
                _override_mask,
                with_indices=True,
            )
            
            # 3)  Now update your index pools:
            #    — add only the good ones to the training set
            current_indices.extend(good_indices)
            #    — put *only* the bad ones back into the unlabeled pool
            #      (everything else except good_indices stays out of the pool permanently)
            remaining_indices = [r for r in remaining_indices if r not in new_indices] + bad_indices

            logger.info(f"Added {len(good_indices)} auto-labeled samples, kept {len(bad_indices)} still unlabeled.")
            
        else:
            # Create the training dataset FOR THIS ITERATION
            current_train_subset = full_dataset_dict['train'].select(current_indices)
            
        logger.info("Training subset preprocessing complete.")

        # --- 6b. Setup Model and Trainer for Current Iteration ---
        # Load model: Start fresh or from previous iteration's checkpoint
        if current_model is None: # First iteration
            logger.info("Loading initial model...")
            current_model = load_model_for_segmentation(
                model_name_or_path=config['model']['name'],
                num_labels=NUM_PASCAL_VOC_LABELS,
                id2label=PASCAL_VOC_ID2LABEL,
                label2id=PASCAL_VOC_LABEL2ID,
                ignore_mismatched_sizes=config['model'].get('ignore_mismatched_sizes', False)
            )
        else:
            logger.info("Re-using model from previous iteration.")
            # Optional: Re-initialize parts of the model if desired between steps
            # e.g., re-init classifier layer? Requires careful consideration.
            
        
        # Define output dir and run name for THIS iteration
        iter_output_dir = f"{output_dir_prefix}_iter_{target_percentage_int}"
        iter_run_name = f"{run_name_prefix}_{target_percentage_int}pct"
        os.makedirs(iter_output_dir, exist_ok=True)

        # Setup W&B Run for this iteration (reinit=True is important)
        # Pass the iteration-specific config/info if needed
        iteration_config = copy.deepcopy(config) # Avoid modifying original config
        iteration_config['active_learning']['current_percentage'] = current_data_percentage
        iteration_config['active_learning']['current_step'] = current_al_step
        setup_wandb(iteration_config, run_name=iter_run_name, project_name=config.get('project_name'))

        # Configure Training Arguments for THIS iteration
        # Make a copy to avoid side effects if looping TrainingArguments object
        iter_training_args = TrainingArguments(
            output_dir=str(iter_output_dir),
            run_name=str(iter_run_name),

            # ints
            num_train_epochs=int(config['training']['num_train_epochs']),
            per_device_train_batch_size=int(config['training']['per_device_train_batch_size']),
            per_device_eval_batch_size=int(config['training']['per_device_eval_batch_size']),
            save_total_limit=int(config['training']['save_total_limit']),
            logging_steps=int(config['training']['logging_steps']),
            seed=int(config['seed']),

            # floats
            learning_rate=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay']),

            # strings
            eval_strategy=str(config['training']['evaluation_strategy']),
            save_strategy=str(config['training']['save_strategy']),
            metric_for_best_model=str(config['training']['metric_for_best_model']),

            # booleans
            load_best_model_at_end=bool(config['training']['load_best_model_at_end']),
            remove_unused_columns=bool(config['training'].get('remove_unused_columns', False)),
            fp16=bool(config['training'].get('fp16', False) and torch.cuda.is_available()),

            # logging / reporting
            logging_dir=os.path.join(iter_output_dir, 'logs'),
            logging_first_step=True,
            report_to=[x.strip() for x in str(config.get('log_with', 'none')).split(',') if x.strip()],

            # never push these interim checkpoints
            push_to_hub=False
        )

        # Initialize compute_metrics function (needed for Trainer)
        compute_metrics_fn = partial(
            compute_metrics_segmentation,
            num_labels=NUM_PASCAL_VOC_LABELS,
            ignore_index=255
        )

        # AL callbacks
        al_progress_callback = ActiveLearningProgressCallback(
            total_al_steps=total_al_steps,
            current_al_step=current_al_step,
            current_data_percentage=current_data_percentage
        )
        image_logger_callback = None

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=config['training'].get('early_stopping_patience', 3),
            early_stopping_threshold=config['training'].get('early_stopping_threshold', 0.0),
        )

        # Initialize Trainer for THIS iteration
        trainer = Trainer(
            model=current_model, # Pass the current model object
            args=iter_training_args,
            train_dataset=current_train_subset,
            eval_dataset=full_dataset_dict['validation'], # Use the FIXED validation set
            compute_metrics=compute_metrics_fn,
            callbacks=[al_progress_callback, early_stopping_callback, image_logger_callback] if image_logger_callback else [al_progress_callback, early_stopping_callback],
            # No need to pass model_init if we reuse the model object
        )
        if image_logger_callback:
            image_logger_callback.trainer = trainer # Set trainer for the callback
            image_logger_callback.train_dataset = current_train_subset # Set train dataset for logging
            image_logger_callback.eval_dataset = full_dataset_dict['validation'] # Set eval dataset for logging
        
        if config.get('debug', False):
            logger.info("Debugging mode enabled. Logging a batch of data.")
            # Log a batch of data
            train_loader = trainer.get_train_dataloader()
            batch = next(iter(train_loader))
            imgs = batch["pixel_values"]   # tensor [B,3,H,W]
            msks = batch["labels"]         # tensor [B,H,W]
            debug_log_and_plot(imgs, msks, PASCAL_VOC_LABEL_NAMES)

        # --- 6c. Train Model for Current Iteration ---
        logger.info(f"Starting training for {target_percentage_int}% data...")
        try:
             trainer.train(
                 # resume_from_checkpoint= 'path/to/checkpoint' # Can be used if needed
                 # If Trainer reuses model object, weights are automatically carried over
             )
             logger.info(f"Training finished for {target_percentage_int}% data.")
        except Exception as e:
             logger.error(f"Training failed at iteration {current_al_step} ({target_percentage_int}%): {e}", exc_info=True)
             # TODO: Decide whether to continue or stop the loop
             break # Stop the loop on training failure

        # --- 6d. Evaluate and Log Metrics for Current Iteration ---
        logger.info(f"Evaluating model on FIXED validation set (after {target_percentage_int}% training)...")
        eval_metrics = trainer.evaluate(eval_dataset=full_dataset_dict['validation'])
        logger.info(f"Validation Metrics ({target_percentage_int}% data): {eval_metrics}")

        # Store metrics for overall summary
        # Ensure keys match what compute_metrics returns (e.g., 'eval_mean_iou')
        metrics_to_store = {k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))}
        overall_metrics[target_percentage_int] = metrics_to_store

        # Save the model checkpoint from this iteration (Trainer does this based on save_strategy)
        # The best model (according to load_best_model_at_end) is kept in trainer.model
        current_model = trainer.model # Update current_model to the best one from this iter

        # Finish W&B run for this iteration ONLY if it's active
        if config.get('log_with') == 'wandb' and trainer.is_world_process_zero():
             import wandb
             # Log the eval metrics explicitly to the *current* run before finishing
             wandb.log({f"final_eval_{k}": v for k, v in eval_metrics.items()})
             wandb.finish() # Finish the per-iteration run

    # --- 7. Final Evaluation on Test Set (using the model from the last iteration) ---
    if current_model and full_dataset_dict['test']:
        logger.info("\n--- Final Evaluation on FIXED Test Set ---")
        # Need a final Trainer instance for evaluation if the last one is out of scope
        # Or just use the last trainer instance if available
        # Re-create args pointing to a final output dir if needed
        final_output_dir = os.path.join(output_dir_prefix, "final")
        os.makedirs(final_output_dir, exist_ok=True)

        final_eval_args = TrainingArguments(
            output_dir=final_output_dir,
            per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
            fp16=config['training'].get('fp16', False) and torch.cuda.is_available(),
            report_to="none", # No need to report this trainer's run separately
            remove_unused_columns=False,
        )
        final_trainer = Trainer(
            model=current_model, # Use the final best model from the loop
            args=final_eval_args,
            eval_dataset=full_dataset_dict['test'],
            compute_metrics=compute_metrics_fn,
        )
        test_metrics = final_trainer.evaluate(eval_dataset=full_dataset_dict['test'])
        logger.info(f"Final Test Set Metrics: {test_metrics}")

        # Save final model and test metrics
        final_trainer.save_model(os.path.join(final_output_dir, "final_model"))
        final_trainer.save_metrics("eval", test_metrics) # Saves to final_output_dir/test_results.json
        logger.info(f"Final model saved to {os.path.join(final_output_dir, 'final_model')}")

        # Log final test metrics to a summary W&B run (optional)
        if config.get('log_with') == 'wandb':
            # Start a new summary run or log to the last iteration's run?
            # Let's log to a new summary run for clarity
            summary_run = setup_wandb(config, run_name=f"{run_name_prefix}_summary", project_name=config.get('project_name'))
            if summary_run:
                wandb.log({"final_test_metrics": test_metrics})
                # Log the overall AL progress summary table/plots
                log_active_learning_summary(overall_metrics, config)
                wandb.finish()


    elif not current_model:
         logger.error("Active learning loop did not produce a final model. Skipping final evaluation.")
    else:
         logger.warning("No test set available. Skipping final evaluation.")

    logger.info("Active learning simulation script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Active Learning Simulation for Image Segmentation")
    parser.add_argument(
        "--config",
        type=str,
        # required=True,
        default='configs/active_learning_config.yaml',
        metavar="CONFIG",
        help="Path to the active learning configuration YAML file."
    )
    args = parser.parse_args()

    # Setup basic logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    run_active_learning_pipeline(args.config)