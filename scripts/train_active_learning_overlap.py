import os
import sys
import argparse
import logging
import math
import random
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="datasets")
warnings.filterwarnings("ignore", category=FutureWarning, module="datasets")
# warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="accelerate")
from functools import partial
from typing import Dict, Any, List
import copy # To deep copy config for modifications per iteration
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    TrainingArguments,
    Trainer,
    SegformerImageProcessor,
    set_seed,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict, concatenate_datasets
from torchmetrics.segmentation import MeanIoU # type: ignore

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
    NUM_PASCAL_VOC_LABELS,
    PASCAL_VOC_BINARY_ID2LABEL,
    PASCAL_VOC_BINARY_LABEL2ID,
    NUM_PASCAL_VOC_BINARY_LABELS
)
from models.segformer import load_model_for_segmentation

def sanity_check_percentages(current_percentage: float, increment: float, max_percentage: float) -> tuple[float, float, float]:
    """Perform sanity checks on the percentage values."""
    # Ensure percentages are in the range [0, 100]
    if current_percentage < 1.0:
        logger.debug(f"Converting current_percentage from {current_percentage} to {current_percentage * 100}")
        current_percentage *= 100
    if increment < 1.0:
        logger.debug(f"Converting increment from {increment} to {increment * 100}")
        increment *= 100
    if max_percentage < 100:
        logger.debug(f"Converting max_percentage from {max_percentage} to {max_percentage * 100}")
        max_percentage *= 100
    if current_percentage > max_percentage:
        raise ValueError(f"Current percentage ({current_percentage}) cannot exceed max percentage ({max_percentage}).")
    if max_percentage > 100:
        raise ValueError("Max percentage cannot exceed 100.")
    if increment <= 0:
        raise ValueError("Increment percentage must be positive.")
    if current_percentage <= 0:
        raise ValueError("Initial percentage must be positive.")
    if max_percentage <= 0:
        raise ValueError("Max percentage must be positive.")
    if current_percentage > 100:
        raise ValueError("Initial percentage cannot exceed 100.")
    
    return current_percentage, increment, max_percentage

def run_active_learning_pipeline(config_path: str):
    """Main function for the active learning simulation script."""
        # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    al_config = config['active_learning']
    dataset_config = config['dataset']
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
    train_indices, val_indices = create_train_val_test_splits(
        raw_dataset['train'],
        # raw_dataset, # here, load_dataset returns a DatasetDict instead of a Dataset
        val_percentage=config['dataset'].get('val_split_percentage', 0.1),
        seed=config['seed'] # Use the global seed for splitting
    )

    # --- 4. Prepare Image Processor and Preprocessing Function ---
    # Load image processor ONCE
    logger.info("Loading Image Processor...")
    image_processor = SegformerImageProcessor.from_pretrained(
        config['dataset']['feature_extractor_name'],
        do_reduce_labels=False
    )
    
    binary_segmentation_task = dataset_config.get('binary_segmentation', False)
    if binary_segmentation_task:
        num_labels = NUM_PASCAL_VOC_BINARY_LABELS
        id2label = PASCAL_VOC_BINARY_ID2LABEL
        label2id = PASCAL_VOC_BINARY_LABEL2ID
    else:
        num_labels = NUM_PASCAL_VOC_LABELS
        id2label = PASCAL_VOC_ID2LABEL
        label2id = PASCAL_VOC_LABEL2ID

    logger.info(f"Binary segmentation task: {binary_segmentation_task}")

    preprocess_fn = partial(
        preprocess_data, 
        image_processor=image_processor,
        image_col=dataset_config['image_col'],
        mask_col=dataset_config['mask_col'],
        binary_segmentation_task=binary_segmentation_task  # Pass the new parameter
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
    
    test_dataset = raw_dataset['test'].map(
        preprocess_fn,
        batched=True,
        batch_size=config['dataset'].get('batch_size', 16),
        remove_columns=[config['dataset']['image_col'], config['dataset']['mask_col']],
        load_from_cache_file=True,
    )
    
    # Split the dataset into train, validation, and test sets
    full_dataset_dict = DatasetDict({
        'train': full_dataset.select(train_indices),
        'validation': full_dataset.select(val_indices),
    })
    
    # --- 5. Initialize Active Learning Loop Variables (already done) ---
    overall_metrics = {}  # Store metrics per iteration
    num_total_train = len(full_dataset_dict['train'])
    logger.info(f"Total training samples: {num_total_train}")
    all_train_indices = list(range(num_total_train))
    random.shuffle(all_train_indices)  # Consistent random sampling

    # Initial AL config parameters
    current_percentage, increment, max_percentage = sanity_check_percentages(
        al_config['initial_percentage'], 
        al_config['increment_percentage'], 
        al_config['max_percentage']
    )

    num_initial = math.ceil(num_total_train * (current_percentage/100.0))
    current_indices = all_train_indices[:num_initial]
    remaining_indices = all_train_indices[num_initial:]

    empty_iterations_count = 0
    iteration = 1
    current_model = None  # Model will be loaded in the loop
    good_indices, bad_indices = None, None
    current_data_percentage = None
    miou_metric = MeanIoU(
        num_classes=NUM_PASCAL_VOC_LABELS, 
        per_class=False, 
        include_background=False, 
        input_format="index"
    )
    added = num_initial
    # --- 6. Active Learning Loop (while loop) ---
    while remaining_indices and empty_iterations_count < 2 and added > 5:
        logger.info(f"\n--- Active Learning Iteration {iteration} ---")
        
        if iteration > 1:
            uncertainties, segmentations = compute_image_uncertainties(
                model=current_model,
                dataset=full_dataset_dict['train'],
                remaining_indices=remaining_indices,
                preprocess_fn=preprocess_fn,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=al_config.get("inference_batch_size", 8),
                # return_tensor=True,
                return_tensor=False,
            )

            # Pick top-k indices by entropy (adjust reverse as needed)
            new_indices = sorted(
                uncertainties, 
                key=lambda idx: uncertainties[idx], 
                reverse=False if al_config.get("min_entropy", False) else True
            # )[:k] # use all indices for now
            )

            # Threshold using mean IoU
            iou_thresh = al_config.get("iou_threshold", 0.8)
            good_indices = []
            bad_indices  = []
            for idx in tqdm(new_indices, desc="Evaluating mIoU", unit="sample"):
                true_mask = full_dataset_dict["train"][idx]["labels"]  # numpy array [H,W]
                pred_mask = segmentations[idx]
                if type(pred_mask) != torch.Tensor:
                    pred_mask_cp = pred_mask.copy()  # numpy array [H,W]
                    miou = compute_mean_iou(true_mask, pred_mask_cp)
                elif type(pred_mask) == torch.Tensor:
                    pred_mask_cp = pred_mask.clone() # torch tensor for torchmetrics miou
                    miou = miou_metric(
                        pred_mask, 
                        true_mask
                    )
                else:
                    raise ValueError(f"Unknown type for pred_mask: {type(pred_mask)}")
                
                if miou >= iou_thresh:
                    good_indices.append(idx) 
                else:
                    bad_indices.append(idx)

            logger.info(f"{len(good_indices)}/{len(new_indices)} passed IoU ≥ {iou_thresh}")

            if good_indices:
                # Override (or inject) the labels (pseudo masks) in place for good samples:
                def _override_mask(example, example_idx):
                    if example_idx in good_indices:
                        return {"labels": segmentations[example_idx].tolist()}
                    return {}
                full_dataset_dict["train"] = full_dataset_dict["train"].map(
                    _override_mask,
                    with_indices=True,
                )
                # Update the pools
                current_indices.extend(good_indices)
                # Remove all evaluated indices from the unlabeled pool and add back the bad ones for re-evaluation
                remaining_indices = [r for r in remaining_indices if r not in new_indices] + bad_indices
                empty_iterations_count = 0
                logger.info(f"Added {len(good_indices)} auto-labeled samples; remaining unlabeled: {len(remaining_indices)}")
            else:
                empty_iterations_count += 1
                logger.info("No new samples met the IoU threshold in this iteration.")
                
        added = len(good_indices) if good_indices else num_initial
        logger.info(f"Added {added} samples in this iteration.")
                
        current_percentage = min(max_percentage, (len(current_indices) / num_total_train) * 100.0)
        logger.info(f"Current data percentage: {current_percentage:.2f} ({len(current_indices)} samples)")
        current_data_percentage = current_percentage / 100.0

        # --- Prepare Training Subset for This Iteration ---
        current_train_subset = full_dataset_dict['train'].select(current_indices)
        logger.info("Training subset preprocessing complete.")

        # --- 6b. Setup Model and Trainer for This Iteration ---
        if current_model is None:  # First iteration
            logger.info("Loading initial model...")
            current_model = load_model_for_segmentation(
                model_name_or_path=config['model']['name'],
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=config['model'].get('ignore_mismatched_sizes', False)
            )
        else:
            logger.info("Re-using model from previous iteration.")
        
        iter_output_dir = f"{output_dir_prefix}_iter_{current_percentage}"
        iter_run_name = f"{run_name_prefix}_{current_percentage}pct"
        os.makedirs(iter_output_dir, exist_ok=True)
        
        # Setup W&B Run for this iteration (if applicable)
        iteration_config = copy.deepcopy(config)
        iteration_config['active_learning']['current_percentage'] = (len(current_indices) / num_total_train) * 100.0 if iteration < 2 else current_data_percentage
        iteration_config['active_learning']['current_step'] = iteration
        setup_wandb(iteration_config, run_name=iter_run_name, project_name=config.get('project_name'))

        # Configure TrainingArguments (same as before)
        iter_training_args = TrainingArguments(
            output_dir=str(iter_output_dir),
            run_name=str(iter_run_name),
            num_train_epochs=int(config['training']['num_train_epochs']),
            per_device_train_batch_size=int(config['training']['per_device_train_batch_size']),
            per_device_eval_batch_size=int(config['training']['per_device_eval_batch_size']),
            save_total_limit=int(config['training']['save_total_limit']),
            logging_steps=int(config['training']['logging_steps']),
            seed=int(config['seed']),
            learning_rate=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay']),
            eval_strategy=str(config['training']['evaluation_strategy']),
            save_strategy=str(config['training']['save_strategy']),
            metric_for_best_model=str(config['training']['metric_for_best_model']),
            load_best_model_at_end=bool(config['training']['load_best_model_at_end']),
            remove_unused_columns=bool(config['training'].get('remove_unused_columns', False)),
            fp16=bool(config['training'].get('fp16', False) and torch.cuda.is_available()),
            logging_dir=os.path.join(iter_output_dir, 'logs'),
            logging_first_step=True,
            report_to=[x.strip() for x in str(config.get('log_with', 'none')).split(',') if x.strip()],
            push_to_hub=False
        )

        compute_metrics_fn = partial(
            compute_metrics_segmentation,
            num_labels=NUM_PASCAL_VOC_LABELS,
            ignore_index=255
        )

        al_progress_callback = ActiveLearningProgressCallback(
            total_al_steps=None,  # Not applicable anymore
            current_al_step=iteration,
            current_data_percentage=current_data_percentage,
            number_of_samples=len(current_train_subset),
            total_number_of_samples=num_total_train,
        )
        
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=config['training'].get('early_stopping_patience', 3),
            early_stopping_threshold=config['training'].get('early_stopping_threshold', 0.0),
        )

        trainer = Trainer(
            model=current_model,
            args=iter_training_args,
            train_dataset=current_train_subset,
            eval_dataset=full_dataset_dict['validation'],
            compute_metrics=compute_metrics_fn,
            callbacks=[al_progress_callback, early_stopping_callback],
        )
        
        # --- 6c. Train and Evaluate This Iteration ---
        logger.info(f"Starting training for {current_percentage}% data with {len(current_indices)} samples...")
        try:
            trainer.train()
            logger.info(f"Training finished for iteration {iteration} at {current_percentage}% data.")
        except Exception as e:
            logger.error(f"Training failed at iteration {iteration} ({current_percentage}%): {e}", exc_info=True)
            break

        eval_metrics = trainer.evaluate(eval_dataset=full_dataset_dict['validation'])
        logger.info(f"Validation Metrics ({current_percentage}% data): {eval_metrics}")
        overall_metrics[current_percentage] = {k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))}
        
        current_model = trainer.model  # Update model

        if config.get('log_with') == 'wandb' and trainer.is_world_process_zero():
            import wandb
            wandb.log({f"final_eval_{k}": v for k, v in eval_metrics.items()})
            wandb.finish()
        
        iteration += 1

    # --- 7. Final Evaluation on Test Set (using the model from the last iteration) ---
    if current_model and test_dataset:
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
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics_fn,
        )
        test_metrics = final_trainer.evaluate(eval_dataset=test_dataset)
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