import os
import sys
import argparse
import logging
import math
import random
from functools import partial
import copy
from typing import List, Dict, Any

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import (
    TrainingArguments,
    Trainer,
    SegformerImageProcessor,
    set_seed,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict, concatenate_datasets

from utils.config import load_config
from utils.log_utils import setup_wandb, logger, log_active_learning_summary
from utils.metrics import compute_metrics_segmentation
from utils.active_learning import (
    sample_initial_data,
    select_next_batch_indices,
    ActiveLearningProgressCallback,
    log_vlm_iteration_summary, # Import VLM summary logger
    feature_extractor_fn
)
from utils.vlm import get_vlm_handler, VLMHandler # Import VLM factory and base class
from data.pascal_voc import (
    load_pascal_voc_dataset,
    preprocess_data,
    create_train_val_test_splits,
    PASCAL_VOC_ID2LABEL,
    PASCAL_VOC_LABEL2ID,
    NUM_PASCAL_VOC_LABELS
)
from models.segformer import load_model_for_segmentation

# --- Helper function for VLM Feedback Simulation ---
def simulate_vlm_feedback_on_batch(
    vlm_handler: VLMHandler,
    model: torch.nn.Module,
    batch: Dict[str, Any], # Batch DIRECTLY from the raw dataset (before preprocessing)
    image_processor: SegformerImageProcessor,
    config: Dict[str, Any],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Gets predictions for a raw batch, simulates VLM feedback on each item.

    Args:
        vlm_handler (VLMHandler): Initialized VLM handler.
        model (torch.nn.Module): The current segmentation model.
        batch (Dict[str, Any]): Batch containing raw 'image' and 'mask'.
        image_processor (SegformerImageProcessor): For preprocessing.
        config (Dict[str, Any]): Experiment configuration (for VLM settings).
        device (torch.device): Device for model inference.

    Returns:
        List[Dict[str, Any]]: List of feedback dictionaries, one per sample in batch.
    """
    images = batch[config['dataset']['image_col']] # List of PIL Images
    gt_masks = batch[config['dataset']['mask_col']] # List of PIL Masks (ground truth)
    feedback_list = []
    vlm_query_template = config.get('vlm_itl', {}).get('vlm_query_template', "Is this segmentation correct?")


    # Preprocess batch for model inference
    preprocess_fn = partial(
        preprocess_data,
        image_processor=image_processor,
        image_col=config['dataset']['image_col'],
        mask_col=config['dataset']['mask_col']
    )
    # Need to handle single items if batch size is 1, map works on lists
    processed_inputs = preprocess_fn({'image': images, 'mask': gt_masks})
    pixel_values = []
    for i in range(len(processed_inputs['pixel_values'])):
        pixel_values.append(processed_inputs['pixel_values'][i].unsqueeze(0).to(device))
    pixel_values = torch.cat(pixel_values, dim=0) # Shape: (batch, 3, H, W)

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits # Shape: (batch, num_labels, H/4, W/4)

    # Upsample logits to match input image size (or label size if different)
    # Note: This assumes labels match the image size after processor's transforms.
    # If image_processor resizes labels differently, adjust target size.
    # Let's assume target size matches input image size for simplicity in VLM context.
    # HACK: Get expected label size from image_processor if possible, else use a default?
    #       Let's assume we need original image size for VLM usually.
    #       The logits are H/4, W/4. We need to resize them.
    original_height, original_width = images[0].size[1], images[0].size[0]

    # Upsample logits to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )
    predicted_mask_tensor = upsampled_logits.argmax(dim=1) # Shape: (batch, H, W)

    # Iterate through batch items for VLM feedback
    for i in range(len(images)):
        original_image = images[i]
        gt_mask_np = processed_inputs["labels"][i].cpu().numpy().astype(np.uint8)
        gt_mask_pil = gt_masks[i]
        
        pred_mask_np = predicted_mask_tensor[i].cpu().numpy().astype(np.uint8)
        pred_mask_pil = Image.fromarray(pred_mask_np, mode='L')
        
        # Find dominant predicted label (excluding background=0 and ignore=255)
        valid_pred = (pred_mask_np != 255) & (pred_mask_np != 0)
        pred_labels, pred_counts = np.unique(
            pred_mask_np[valid_pred],
            return_counts=True
        )
        dominant_pred_label_id = pred_labels[np.argmax(pred_counts)] if len(pred_labels) > 0 else -1

        # Find dominant ground truth label (excluding background=0 and ignore=255)
        valid_gt = (gt_mask_np != 255) & (gt_mask_np != 0)
        gt_labels, gt_counts = np.unique(
            gt_mask_np[valid_gt],
            return_counts=True
        )
        dominant_gt_label_id = gt_labels[np.argmax(gt_counts)] if len(gt_labels) > 0 else -1
        predicted_label_name = PASCAL_VOC_ID2LABEL.get(dominant_pred_label_id, "unknown")
        ground_truth_label_name = PASCAL_VOC_ID2LABEL.get(dominant_gt_label_id, "unknown")

        # Get feedback from VLM handler
        feedback = vlm_handler.get_vlm_feedback(
            image=original_image,
            segmentation_mask=pred_mask_pil, # Provide the predicted mask visualization
            predicted_label_name=predicted_label_name,
            ground_truth_label_name=ground_truth_label_name,
            query_template=vlm_query_template
        )
        feedback_list.append(feedback)

    model.train() # Set model back to train mode
    return feedback_list


# --- Main Pipeline ---
def run_vlm_itl_pipeline(config_path: str):
    """Main function for the VLM-In-The-Loop active learning simulation."""
    # --- 1. Load Configuration ---
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Check if VLM-ITL is enabled in the config (can reuse AL config)
    if not config.get('vlm_itl', {}).get('enabled', False):
        logger.error("VLM-ITL section not enabled in the configuration file. Set 'vlm_itl.enabled = True'.")
        sys.exit(1)

    al_config = config['active_learning']
    vlm_config = config['vlm_itl']
    # Use VLM-specific prefixes if provided, otherwise fallback to AL prefixes
    run_name_prefix = vlm_config.get('run_name_prefix', config.get('run_name_prefix', 'vlm_itl_run'))
    output_dir_prefix = vlm_config.get('output_dir_prefix', config.get('output_dir_prefix', './results/vlm_itl'))

    # --- 2. Setup Seed and VLM Handler ---
    set_seed(config['seed'])
    logger.info(f"Global seed set to {config['seed']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Initializing VLM Handler...")
    try:
        vlm_handler = get_vlm_handler(config)
        vlm_handler._load_model()  # Load the VLM model
        logger.info(f"Using VLM handler: {type(vlm_handler).__name__}")
    except Exception as e:
        logger.error(f"Failed to initialize VLM handler: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Load and Prepare Full Data (Same as Active Learning baseline) ---
    logger.info("Loading PASCAL VOC dataset...")
    raw_datasets = load_pascal_voc_dataset(
        dataset_name=config['dataset']['name'],
        cache_dir=config['dataset'].get('cache_dir')
    )
    logger.info("Creating fixed validation and test sets...")
    full_train_data, val_dataset, test_dataset = create_train_val_test_splits(
        raw_datasets['train'],
        val_percentage=config['dataset'].get('val_split_percentage', 0.1),
        test_percentage=config['dataset'].get('test_split_percentage', 0.1),
        seed=config['seed']
    )
    logger.info(f"Full Train Data size: {len(full_train_data)}, Val Set size: {len(val_dataset)}, Test Set size: {len(test_dataset)}")

    # --- 4. Prepare Image Processor and Preprocessing Function ---
    logger.info("Loading Image Processor...")
    image_processor = SegformerImageProcessor.from_pretrained(
        config['dataset']['feature_extractor_name'],
        do_reduce_labels=False
    )
    preprocess_fn = partial(
        preprocess_data, image_processor=image_processor,
        image_col=config['dataset']['image_col'], mask_col=config['dataset']['mask_col']
    )
    logger.info("Preprocessing fixed validation and test sets...")
    processed_val_dataset = val_dataset.map(
        preprocess_fn, 
        batched=True, 
        remove_columns=val_dataset.column_names,
        batch_size=config['training'].get('per_device_eval_batch_size', 8), # Use eval batch size for preprocessing
        load_from_cache_file=False # Force reprocessing to avoid cache issues
        )
    processed_test_dataset = test_dataset.map(
        preprocess_fn, 
        batched=True, 
        remove_columns=test_dataset.column_names,
        batch_size=config['training'].get('per_device_eval_batch_size', 8), # Use eval batch size for preprocessing
        load_from_cache_file=False # Force reprocessing to avoid cache issues)
    )
    processed_val_dataset.set_format("torch")
    processed_test_dataset.set_format("torch")

    # --- 5. Initialize Active Learning Loop Variables ---
    overall_metrics = {}
    all_vlm_feedback = {} # Store VLM feedback per iteration {data_percentage: feedback_list}
    num_total_train = len(full_train_data)
    all_train_indices = list(range(num_total_train))
    random.seed(config['seed'])
    random.shuffle(all_train_indices)

    current_percentage = al_config['initial_percentage']
    increment = al_config['increment_percentage']
    max_percentage = al_config['max_percentage']
    num_initial = math.ceil(num_total_train * current_percentage)
    current_indices = all_train_indices[:num_initial]
    remaining_indices = all_train_indices[num_initial:]

    al_steps = []
    p = current_percentage
    while p <= max_percentage + 1e-6:
        al_steps.append(int(round(p * 100)))
        p += increment
    total_al_steps = len(al_steps)
    logger.info(f"Planned Active Learning percentages: {al_steps}%")

    # --- 6. Active Learning Loop with VLM Feedback ---
    current_model = None
    for i, target_percentage_int in enumerate(al_steps):
        current_al_step = i + 1
        current_data_percentage = target_percentage_int / 100.0

        # --- 6a. Prepare Data for Current Iteration (Same as AL baseline) ---
        # num_target_samples = math.ceil(num_total_train * current_data_percentage)
        # num_current_samples = len(current_indices)
        # num_to_add = max(0, num_target_samples - num_current_samples)

        # logger.info(f"\n--- VLM-ITL Iteration {current_al_step}/{total_al_steps} ---")
        # logger.info(f"Target Data: {target_percentage_int}% ({num_target_samples} samples)")

        # if num_to_add > 0 and remaining_indices:
        #     num_to_select = min(num_to_add, len(remaining_indices))
        #     logger.info(f"Selecting {num_to_select} new samples (using '{al_config.get('sampling_strategy', 'random')}' strategy)...")
        #     new_indices = select_next_batch_indices(
        #         remaining_indices, num_to_select, strategy=al_config.get("sampling_strategy", "random")
        #     )
        #     current_indices.extend(new_indices)
        #     new_indices_set = set(new_indices)
        #     remaining_indices = [idx for idx in remaining_indices if idx not in new_indices_set]
        #     logger.info(f"Total training samples now: {len(current_indices)}")

        # current_train_subset_raw = full_train_data.select(current_indices)
        # logger.info(f"Preprocessing training subset ({len(current_train_subset_raw)} samples)...")
        # current_train_subset_processed = current_train_subset_raw.map(
        #     preprocess_fn, 
        #     batched=True, 
        #     remove_columns=current_train_subset_raw.column_names,
        #     batch_size=config['training'].get('per_device_train_batch_size', 8), # Use train batch size for preprocessing
        #     load_from_cache_file=False # Force reprocessing to avoid cache issues
        # )
        # current_train_subset_processed.set_format("torch")
        
        # --- 6a. Propose & Pseudo-Label with VLM ---
        num_target_samples  = math.ceil(num_total_train * current_data_percentage)
        num_current_samples = len(current_indices)
        num_to_add          = max(0, num_target_samples - num_current_samples)
        logger.info(f"\n--- VLM-ITL Iteration {current_al_step}/{total_al_steps} ---")
        if num_to_add > 0 and remaining_indices:
            num_to_select = min(num_to_add, len(remaining_indices))
            logger.info(f"Proposing {num_to_select} new samples for pseudo-labeling using "
                        f"'{al_config.get('sampling_strategy','random')}' strategy…")

            # 1) Propose a batch from the unlabeled pool
            proposals = select_next_batch_indices(
                remaining_indices,
                num_to_select,
                strategy=al_config.get("sampling_strategy", "random"),
                model=current_model,
                dataset=full_train_data,
                preprocess_fn=preprocess_fn,
                feature_extractor_fn=feature_extractor_fn,
                device=device,
                mc_iterations=al_config.get("mc_iterations", 5),
                diversify_pool_factor=al_config.get("diversify_pool_factor", 10),
            )

            # 2) Verify each proposal with the VLM
            accepted, rejected = [], []
            per_device = config['training']['per_device_eval_batch_size']
            for start in range(0, len(proposals), per_device):
                batch_idxs = proposals[start : start + per_device]
                batch_raw = full_train_data.select(batch_idxs)
                feedback = simulate_vlm_feedback_on_batch(
                    vlm_handler,
                    current_model,
                    batch_raw,
                    image_processor,
                    config,
                    device
                )
                # collect those the VLM “agrees” on
                for idx, fb in zip(batch_idxs, feedback):
                    if fb["vlm_agrees_with_gt"]:
                        accepted.append(idx)
                    else:
                        rejected.append(idx)

            logger.info(f"VLM accepted {len(accepted)} samples, rejected {len(rejected)}")

            # 3) Update labeled / unlabeled pools
            current_indices.extend(accepted)
            remaining_indices = [idx for idx in remaining_indices if idx not in accepted]

        elif not remaining_indices and num_to_add > num_current_samples:
            logger.warning("No more remaining indices to sample from, but target size not reached.")
        
        logging.warning(f"type of full train: {type(full_train_data)}")
        current_train_subset_raw = full_train_data.select(current_indices)
        logger.info(f"Preprocessing training subset ({len(current_train_subset_raw)} samples)...")
        current_train_subset_processed = current_train_subset_raw.map(
            preprocess_fn, 
            batched=True, 
            remove_columns=current_train_subset_raw.column_names,
            batch_size=config['training'].get('per_device_train_batch_size', 8), # Use train batch size for preprocessing
            load_from_cache_file=False # Force reprocessing to avoid cache issues
        )
        logging.warning(f"type of preprocessed full train: {type(full_train_data)}")
        current_train_subset_processed.set_format("torch")
        logging.warning(f"type of preprocessed full train: {type(full_train_data)}")

        # --- 6b. Setup Model and Trainer (Same as AL baseline) ---
        if current_model is None:
            logger.info("Loading initial model...")
            current_model = load_model_for_segmentation(
                model_name_or_path=config['model']['name'], num_labels=NUM_PASCAL_VOC_LABELS,
                id2label=PASCAL_VOC_ID2LABEL, label2id=PASCAL_VOC_LABEL2ID,
                ignore_mismatched_sizes=config['model'].get('ignore_mismatched_sizes', False)
            ).to(device) # Move model to device
        else:
            logger.info("Re-using model from previous iteration.")
            current_model.to(device) # Ensure model is on correct device

        iter_output_dir = f"{output_dir_prefix}_iter_{target_percentage_int}"
        iter_run_name = f"{run_name_prefix}_{target_percentage_int}pct"
        os.makedirs(iter_output_dir, exist_ok=True)

        iteration_config = copy.deepcopy(config)
        iteration_config['active_learning']['current_percentage'] = current_data_percentage
        iteration_config['active_learning']['current_step'] = current_al_step
        setup_wandb(iteration_config, run_name=iter_run_name, project_name=config.get('project_name'))

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
        compute_metrics_fn = partial(compute_metrics_segmentation, num_labels=NUM_PASCAL_VOC_LABELS, ignore_index=255)
        al_progress_callback = ActiveLearningProgressCallback(total_al_steps, current_al_step, current_data_percentage)
        early_stopping_callback = EarlyStoppingCallback(
            config['training'].get('early_stopping_patience', 3), config['training'].get('early_stopping_threshold', 0.0)
        )
        trainer = Trainer(
            model=current_model, args=iter_training_args,
            train_dataset=current_train_subset_processed, eval_dataset=processed_val_dataset,
            compute_metrics=compute_metrics_fn, callbacks=[al_progress_callback, early_stopping_callback]
        )

        # --- 6c. Train Model (Same as AL baseline) ---
        logger.info(f"Starting training for {target_percentage_int}% data...")
        try:
            trainer.train()
            logger.info(f"Training finished for {target_percentage_int}% data.")
        except Exception as e:
            logger.error(f"Training failed at iteration {current_al_step} ({target_percentage_int}%): {e}", exc_info=True)
            break

        # --- 6d. Evaluate and Log Metrics (Standard Evaluation) ---
        logger.info(f"Evaluating model on FIXED validation set (after {target_percentage_int}% training)...")
        eval_metrics = trainer.evaluate(eval_dataset=processed_val_dataset)
        logger.info(f"Validation Metrics ({target_percentage_int}% data): {eval_metrics}")
        metrics_to_store = {k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))}
        overall_metrics[target_percentage_int] = metrics_to_store

        # --- 6e. Simulate and Log VLM Feedback ---
        # We simulate VLM feedback *after* training for this iteration, using the trained model.
        # This simulates a human (or VLM) reviewing the model's performance at this stage.
        logger.info(f"Simulating VLM feedback on a sample of the validation set...")
        # Use a subset of validation data for VLM feedback simulation to save time/cost
        vlm_eval_subset_size = vlm_config.get('eval_subset_size', 100) # Number of samples for VLM eval
        vlm_eval_subset_size = min(vlm_eval_subset_size, len(val_dataset))
        if vlm_eval_subset_size > 0:
            # Select a random subset of the *raw* validation data
            vlm_eval_indices = random.sample(range(len(val_dataset)), vlm_eval_subset_size)
            vlm_eval_subset_raw = val_dataset.select(vlm_eval_indices)

            iteration_vlm_feedback = []
            # Process in batches for efficiency
            batch_size = config['training']['per_device_eval_batch_size']
            for i in tqdm(range(0, len(vlm_eval_subset_raw), batch_size), desc="VLM Feedback Sim"):
                batch_raw = vlm_eval_subset_raw[i : i + batch_size] # Dict[str, List]
                feedback = simulate_vlm_feedback_on_batch(
                    vlm_handler, current_model, batch_raw, image_processor, config, device
                )
                iteration_vlm_feedback.extend(feedback)

            all_vlm_feedback[target_percentage_int] = iteration_vlm_feedback
            # Log summary stats for this iteration's VLM feedback
            log_vlm_iteration_summary(iteration_vlm_feedback, target_percentage_int)
        else:
             logger.info("Skipping VLM feedback simulation as eval_subset_size is 0.")


        # --- 6f. Update model and finish iteration ---
        current_model = trainer.model # Keep the best model from this iteration
        if config.get('log_with') == 'wandb' and trainer.is_world_process_zero():
            import wandb
            wandb.log({f"final_eval_{k}": v for k, v in eval_metrics.items()})
            # Optionally log VLM summary table for the iteration here if needed
            wandb.finish()

    # --- 7. Final Evaluation on Test Set (Same as AL baseline) ---
    if current_model and processed_test_dataset:
        logger.info("\n--- Final Evaluation on FIXED Test Set ---")
        final_output_dir = os.path.join(output_dir_prefix, "final")
        os.makedirs(final_output_dir, exist_ok=True)
        final_eval_args = TrainingArguments(
            output_dir=final_output_dir, 
            per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
            fp16=config['training'].get('fp16', False) and torch.cuda.is_available(), 
            report_to="none", 
            remove_unused_columns=False,
        )
        final_trainer = Trainer(model=current_model, args=final_eval_args, compute_metrics=compute_metrics_fn)
        test_metrics = final_trainer.evaluate(eval_dataset=processed_test_dataset)
        logger.info(f"Final Test Set Metrics: {test_metrics}")
        final_trainer.save_model(os.path.join(final_output_dir, "final_model"))
        final_trainer.save_metrics("eval", test_metrics)
        logger.info(f"Final model saved to {os.path.join(final_output_dir, 'final_model')}")

        # --- 8. Log Final Summary (including VLM info) ---
        if config.get('log_with') == 'wandb':
            summary_run_name = f"{run_name_prefix}_summary"
            summary_run = setup_wandb(config, run_name=summary_run_name, project_name=config.get('project_name'))
            if summary_run:
                wandb.log({"final_test_metrics": test_metrics})
                log_active_learning_summary(overall_metrics, config) # Log standard AL metrics plot

                # Log overall VLM feedback summary (e.g., agreement rate vs data %)
                if all_vlm_feedback:
                    vlm_summary_metrics = {}
                    for percentage, feedback_list in sorted(all_vlm_feedback.items()):
                        num = len(feedback_list)
                        if num > 0:
                            agree_rate = sum(f['vlm_agrees_with_gt'] for f in feedback_list) / num
                            actual_acc = sum(f['is_segmentation_correct'] for f in feedback_list) / num
                            vlm_summary_metrics[percentage] = {
                                'vlm_agreement_rate': agree_rate,
                                'actual_accuracy_sampled': actual_acc,
                                'vlm_samples_evaluated': num
                            }
                    # Use the existing logger function with a different metrics dict
                    log_active_learning_summary(vlm_summary_metrics, config) # Reuse plotting logic

                wandb.finish()

    elif not current_model:
        logger.error("VLM-ITL loop did not produce a final model. Skipping final evaluation.")
    else:
        logger.warning("No test set available. Skipping final evaluation.")

    logger.info("VLM-ITL simulation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM-In-The-Loop Active Learning Simulation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file (should include vlm_itl settings)."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

    try:
        run_vlm_itl_pipeline(args.config)
    except Exception as e:
        logger.error("An error occurred during the VLM-ITL pipeline.", exc_info=True)
        sys.exit(1)