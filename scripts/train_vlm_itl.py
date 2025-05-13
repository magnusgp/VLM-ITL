import os
import sys
import math
import random
import json
import logging # Ensure logging is imported at the top level
from functools import partial
from typing import Dict, Any, List, Optional # Ensure Optional and List are imported

from PIL import Image
import torch
import numpy as np
from tqdm.auto import tqdm # Added for progress bars

from transformers import (
    TrainingArguments,
    Trainer,
    SegformerImageProcessor,
    set_seed,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict, concatenate_datasets # Ensure Dataset, DatasetDict, concatenate_datasets are imported

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config import load_config
from utils.log_utils import (
    setup_wandb, 
    logger, 
    # log_active_learning_summary, # Not used here
    # debug_log_and_plot # Not used here
)
from utils.metrics import compute_metrics_segmentation
from utils.vlm import get_vlm_handler # Assuming this is where VLM handlers are defined
from utils.active_learning import ( # For SegmentationImageLoggerCallbackVLM if used
    # sample_initial_data, 
    # select_next_batch_indices,
    # feature_extractor_fn,
    # ActiveLearningProgressCallback, 
    SegmentationImageLoggerCallback, # Or a VLM specific version
    # compute_image_uncertainties,
    # compute_mean_iou
)
from data.pascal_voc import (
    load_pascal_voc_dataset,
    preprocess_data,
    create_train_val_test_splits, 
    PASCAL_VOC_LABEL_NAMES,
    PASCAL_VOC_ID2LABEL,
    PASCAL_VOC_LABEL2ID,
    NUM_PASCAL_VOC_LABELS,
    ensure_mask_as_numpy # Make sure this utility is available if used
)
from models.segformer import load_model_for_segmentation

# Conditional import for wandb
try:
    import wandb
except ImportError:
    wandb = None


def calculate_prediction_certainty(
    logits: torch.Tensor, # Upsampled logits (Batch=1, NumClasses, H, W)
    pred_mask_np: np.ndarray, # Single predicted mask (H, W)
    dominant_label_id: Optional[int],
    ignore_index: int = 255
) -> float:
    """
    Calculates certainty: average max probability for pixels of the dominant predicted class.
    If no dominant class, or dominant class has no pixels, returns 0.
    """
    if dominant_label_id is None or dominant_label_id == -1 or dominant_label_id == ignore_index:
        return 0.0

    # Ensure logits are on the correct device, matching pred_mask_np if it becomes a tensor
    # probs = torch.softmax(logits.squeeze(0), dim=0)  # (NumClasses, H, W)
    
    # For numerical stability with upsampled logits, ensure they are float32
    probs = torch.softmax(logits.squeeze(0).float(), dim=0)


    # Probabilities for the dominant class
    # Ensure dominant_label_id is within bounds
    if not (0 <= dominant_label_id < probs.shape[0]):
        logger.warning(f"Dominant label ID {dominant_label_id} is out of bounds for probs shape {probs.shape}. Returning 0 certainty.")
        return 0.0
    dominant_class_probs = probs[dominant_label_id, :, :] # (H, W)
    
    # Mask for pixels belonging to the dominant class in the prediction
    # Ensure pred_mask_np is on the same device as dominant_class_probs if it's converted to a tensor
    dominant_class_pixel_mask = torch.from_numpy(pred_mask_np == dominant_label_id).to(dominant_class_probs.device)
    
    sum_mask = dominant_class_pixel_mask.sum()
    if sum_mask == 0:
        return 0.0 # No pixels predicted as the dominant class
        
    # Average probability for the dominant class over its predicted pixels
    certainty = torch.sum(dominant_class_probs * dominant_class_pixel_mask) / sum_mask
    return certainty.item()


def run_vlm_itl_pipeline(config_path: str):
    logger.info(f"Starting VLM-In-The-Loop pipeline with config: {config_path}")
    config = load_config(config_path)

    # --- 1. Configuration & Setup ---
    if not config.get('vlm_itl', {}).get('enabled', False):
        logger.error("VLM-ITL section not enabled in config. Set 'vlm_itl.enabled = True'.")
        sys.exit(1)

    general_config = config
    vlm_config = config['vlm_itl']
    dataset_config = config['dataset']
    model_config = config['model']
    training_config = config['training']

    set_seed(general_config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() and general_config.get('use_cuda', True) else "cpu")
    logger.info(f"Using device: {device}")
    
    output_dir_base = vlm_config.get('output_dir', './results/vlm_itl_runs')
    run_name_suffix = random.randint(1000,9999)
    run_name = f"{vlm_config.get('run_name_prefix', 'vlm_itl')}_{general_config['seed']}_{run_name_suffix}"
    output_dir = os.path.join(output_dir_base, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Run output will be saved to: {output_dir}")

    if general_config.get('wandb_enabled', False) and wandb:
        setup_wandb(config, run_name, project_name=general_config.get('project_name', 'VLM-ITL-Project'), project_suffix="-vlm_itl")
    else:
        logger.info("W&B logging is disabled.")


    # --- 2. Initialize VLM Handler (BLIP) ---
    logger.info("Initializing VLM Handler (HuggingFace BLIP)...")
    try:
        vlm_handler_config = config.copy() 
        vlm_handler_config['vlm_itl'] = vlm_handler_config.get('vlm_itl', {})
        vlm_handler_config['vlm_itl']['vlm_handler'] = vlm_config.get('vlm_handler_name', 'huggingface_blip')
        vlm_handler = get_vlm_handler(vlm_handler_config)
    except Exception as e:
        logger.error(f"Failed to initialize VLM Handler: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Load Raw Data ---
    logger.info(f"Loading PASCAL VOC dataset: {dataset_config['name']}...")
    raw_datasets = load_pascal_voc_dataset(
        dataset_name=dataset_config['name'],
        cache_dir=dataset_config.get('cache_dir') 
    )
    
    # --- 4. Prepare Image Processor and Preprocessing Function ---
    logger.info("Loading Image Processor...")
    image_processor = SegformerImageProcessor.from_pretrained(
        dataset_config['feature_extractor_name'],
        do_reduce_labels=False 
    )
    
    shared_preprocess_fn = partial(
        preprocess_data, 
        image_processor=image_processor,
        image_col=dataset_config['image_col'],
        mask_col=dataset_config['mask_col']
    )
    
    logger.info("Creating fixed validation and test set indices from raw_datasets['train']...")
    train_indices, val_indices, test_indices = create_train_val_test_splits(
        raw_datasets['train'], 
        val_percentage=dataset_config.get('val_split_percentage', 0.1),
        test_percentage=dataset_config.get('test_split_percentage', 0.1),
        seed=general_config['seed']
    )
    
    logger.info("Preprocessing the entirety of raw_datasets['train'] to create a source for train/val/test splits...")
    preprocessed_source_dataset = raw_datasets['train'].map(
        shared_preprocess_fn,
        batched=True,
        batch_size=training_config.get('per_device_eval_batch_size', 8),
        remove_columns=[dataset_config['image_col'], dataset_config['mask_col']],
        load_from_cache_file=general_config.get('load_from_cache_file', True),
        desc="Preprocessing raw training data"
    )
    preprocessed_source_dataset.set_format("torch", columns=["pixel_values", "labels"])
    
    processed_train_pool = preprocessed_source_dataset.select(train_indices)
    processed_val_dataset = preprocessed_source_dataset.select(val_indices)
    processed_test_dataset = preprocessed_source_dataset.select(test_indices)
    raw_train_data_subset = raw_datasets['train'].select(train_indices)
    
    logger.info(f"Processed train pool size: {len(processed_train_pool)}")
    logger.info(f"Raw train data subset size (for pseudo-labeling source): {len(raw_train_data_subset)}")
    logger.info(f"Processed validation data size: {len(processed_val_dataset)}")
    logger.info(f"Processed test data size: {len(processed_test_dataset)}")

    initial_train_percentage = vlm_config.get('initial_training_percentage', 0.05)
    num_initial_samples = math.ceil(initial_train_percentage * len(processed_train_pool))
    
    all_indices_for_train_pool = list(range(len(processed_train_pool)))
    random.shuffle(all_indices_for_train_pool) 
    
    current_gt_labeled_indices = sorted(all_indices_for_train_pool[:num_initial_samples])
    unlabeled_indices = sorted(all_indices_for_train_pool[num_initial_samples:])
    pseudo_labels_for_indices: Dict[int, Image.Image] = {} 

    logger.info(f"Initial GT labeled set size (from processed pool): {len(current_gt_labeled_indices)}")
    logger.info(f"Initial unlabeled pool size (from processed pool): {len(unlabeled_indices)}")

    # --- 5. VLM Iteration Loop ---
    num_vlm_iterations = vlm_config.get('num_vlm_iterations', 5)
    samples_to_evaluate_certainty_per_iter = vlm_config.get('samples_to_evaluate_certainty_per_iter', 200)
    samples_to_query_vlm_per_iter = vlm_config.get('samples_to_query_vlm_per_iter', 50)
    vlm_query_template = vlm_config.get('vlm_query_template', "Is the primary object in this segmented region a {label_name}?")
    current_model = None

    for iteration in range(num_vlm_iterations):
        logger.info(f"--- VLM Iteration {iteration + 1} / {num_vlm_iterations} ---")
        
        iter_output_dir = os.path.join(output_dir, f"iteration_{iteration + 1}")
        os.makedirs(iter_output_dir, exist_ok=True)
        
        # --- 5.A. Prepare Training Data for Current Iteration ---
        logger.info("Preparing training data for current iteration...")
        gt_indices_in_pool = current_gt_labeled_indices.copy()
        pseudo_indices_in_pool = list(pseudo_labels_for_indices.keys())
        
        iter_train_datasets_to_combine = []

        if not gt_indices_in_pool and not pseudo_indices_in_pool:
            logger.warning("No training data (GT or pseudo) available. Skipping training for this iteration.")
            # Model remains 'current_model' from previous iter or None
        else:
            logger.info(f"Training with {len(gt_indices_in_pool)} GT samples and {len(pseudo_indices_in_pool)} pseudo-labeled samples")
            if gt_indices_in_pool:
                processed_gt_data_for_iter = processed_train_pool.select(gt_indices_in_pool)
                iter_train_datasets_to_combine.append(processed_gt_data_for_iter)
            
            if pseudo_indices_in_pool:
                pseudo_samples_raw_with_new_masks = []
                for idx_in_pool in pseudo_indices_in_pool:
                    original_raw_sample = raw_train_data_subset[idx_in_pool]
                    pseudo_mask_pil = pseudo_labels_for_indices[idx_in_pool]
                    pseudo_samples_raw_with_new_masks.append({
                        dataset_config['image_col']: original_raw_sample[dataset_config['image_col']],
                        dataset_config['mask_col']: pseudo_mask_pil
                    })
                
                temp_pseudo_dataset_raw = Dataset.from_list(pseudo_samples_raw_with_new_masks)
                logger.info(f"Preprocessing {len(temp_pseudo_dataset_raw)} pseudo-labeled samples for iteration {iteration + 1}...")
                processed_pseudo_data_for_iter = temp_pseudo_dataset_raw.map(
                    shared_preprocess_fn,
                    batched=True,
                    batch_size=training_config.get('per_device_train_batch_size', 8),
                    remove_columns=temp_pseudo_dataset_raw.column_names,
                    load_from_cache_file=False, # Crucial for dynamic pseudo-labels
                    desc=f"Preprocessing pseudo-labels for iter {iteration+1}"
                )
                processed_pseudo_data_for_iter.set_format("torch", columns=["pixel_values", "labels"])
                iter_train_datasets_to_combine.append(processed_pseudo_data_for_iter)
            
            processed_iter_train_dataset = None
            if len(iter_train_datasets_to_combine) > 1:
                processed_iter_train_dataset = concatenate_datasets(iter_train_datasets_to_combine).shuffle(seed=general_config['seed'])
            elif iter_train_datasets_to_combine:
                processed_iter_train_dataset = iter_train_datasets_to_combine[0].shuffle(seed=general_config['seed'])
            
            if processed_iter_train_dataset and len(processed_iter_train_dataset) > 0:
                logger.info(f"Final processed training dataset for iteration {iteration+1}: {len(processed_iter_train_dataset)} samples")

                # --- 5.B. Train Segmentation Model ---
                model_path_for_iter = model_config['name']
                if current_model is not None and vlm_config.get('continue_training_from_checkpoint', True):
                    # If we have a model from a previous iteration, we can use its path to continue training
                    # This assumes current_model is the actual model object, not a path.
                    # For continuing training with Trainer, it's often easier to just pass the model object.
                    # If 'current_model' was a path, then model_path_for_iter = current_model
                    logger.info(f"Continuing training from model of previous iteration.")
                    model = current_model # Use the model object directly
                else: # Load fresh or specified base model
                    logger.info(f"Loading segmentation model: {model_path_for_iter} for iteration {iteration+1}")
                    model = load_model_for_segmentation(
                        model_name_or_path=model_path_for_iter,
                        num_labels=NUM_PASCAL_VOC_LABELS,
                        id2label=PASCAL_VOC_ID2LABEL,
                        label2id=PASCAL_VOC_LABEL2ID,
                        ignore_mismatched_sizes=model_config.get('ignore_mismatched_sizes', True)
                    )
                model.to(device)

                iter_training_args_output_dir = os.path.join(iter_output_dir, "training_checkpoints")
                os.makedirs(iter_training_args_output_dir, exist_ok=True)

                iter_training_args = TrainingArguments(
                    output_dir=iter_training_args_output_dir,
                    num_train_epochs=training_config.get('num_train_epochs_per_iter', 3),
                    per_device_train_batch_size=training_config['per_device_train_batch_size'],
                    per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
                    save_strategy=training_config.get("save_strategy_per_iter", "epoch"),
                    evaluation_strategy=training_config.get("evaluation_strategy_per_iter", "epoch"),
                    logging_steps=training_config.get('logging_steps', 50),
                    learning_rate=training_config.get('learning_rate_per_iter', 5e-5),
                    weight_decay=training_config.get('weight_decay', 0.01),
                    metric_for_best_model=training_config.get('metric_for_best_model', 'eval_mean_iou'),
                    load_best_model_at_end=training_config.get('load_best_model_at_end_per_iter', True),
                    remove_unused_columns=False,
                    fp16=training_config.get('fp16', False) and torch.cuda.is_available(),
                    report_to=["wandb"] if general_config.get('wandb_enabled', False) and wandb else ["none"],
                    seed=general_config['seed'],
                    logging_dir=os.path.join(iter_output_dir, 'logs'),
                    disable_tqdm=general_config.get('disable_tqdm', False),
                    push_to_hub=False,
                )

                compute_metrics_fn = partial(
                    compute_metrics_segmentation,
                    num_labels=NUM_PASCAL_VOC_LABELS,
                    ignore_index=255
                )
                
                callbacks = []
                if training_config.get('early_stopping_patience_per_iter', 0) > 0 :
                    callbacks.append(EarlyStoppingCallback(
                        early_stopping_patience=training_config['early_stopping_patience_per_iter'],
                        early_stopping_threshold=training_config.get('early_stopping_threshold_per_iter', 0.0)
                    ))
                
                trainer = Trainer(
                    model=model,
                    args=iter_training_args,
                    train_dataset=processed_iter_train_dataset,
                    eval_dataset=processed_val_dataset,
                    compute_metrics=compute_metrics_fn,
                    callbacks=callbacks if callbacks else None,
                )

                logger.info(f"Starting training for VLM iteration {iteration + 1}...")
                train_result = trainer.train()
                logger.info(f"Training finished for iteration {iteration + 1}. Metrics: {train_result.metrics}")
                
                current_model = trainer.model 
                model_save_path = os.path.join(iter_output_dir, "best_model_from_iter")
                current_model.save_pretrained(model_save_path)
                logger.info(f"Saved model from iteration {iteration+1} to {model_save_path}")


                logger.info(f"Evaluating model from iteration {iteration + 1} on validation set...")
                eval_metrics = trainer.evaluate(eval_dataset=processed_val_dataset)
                logger.info(f"Validation Metrics (Iter {iteration + 1}): {eval_metrics}")
                if general_config.get('wandb_enabled', False) and wandb:
                    wandb.log({f"iter_{iteration+1}_val/{k}": v for k,v in eval_metrics.items()}, step=iteration+1)
            else:
                logger.warning(f"Skipping model training for iteration {iteration + 1} due to no training data.")

        # --- 5.C. Generate Predictions on Unlabeled Pool ---
        if current_model is None:
            logger.error(f"No model available at iteration {iteration+1} to make predictions. Attempting to load base model.")
            current_model = load_model_for_segmentation(
                model_name_or_path=model_config['name'], 
                num_labels=NUM_PASCAL_VOC_LABELS, id2label=PASCAL_VOC_ID2LABEL, label2id=PASCAL_VOC_LABEL2ID,
                ignore_mismatched_sizes=model_config.get('ignore_mismatched_sizes', True)
            ).to(device)
            if current_model is None: # Still None after trying to load
                 logger.error(f"Failed to load base model. Cannot proceed with VLM queries for iter {iteration+1}.")
                 if iteration < num_vlm_iterations -1:
                     continue # Try next iteration, maybe data will appear
                 else:
                     break # End of iterations

        if not unlabeled_indices:
            logger.info("No more unlabeled samples to query. Ending VLM iterations.")
            break

        logger.info(f"Generating predictions for {len(unlabeled_indices)} unlabeled samples...")
        unlabeled_processed_data_for_prediction = processed_train_pool.select(unlabeled_indices)
        unlabeled_raw_data_for_vlm = raw_train_data_subset.select(unlabeled_indices) # For VLM interaction

        pred_args_output_dir = os.path.join(iter_output_dir, f"predictions_iter_{iteration+1}")
        # os.makedirs(pred_args_output_dir, exist_ok=True) # Not strictly needed for TrainingArguments if not saving from it

        pred_args = TrainingArguments(
            output_dir=pred_args_output_dir, # Needs to be a dir
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'] * 2,
            remove_unused_columns=False,
            fp16=training_config.get('fp16', False) and torch.cuda.is_available(),
            report_to="none",
        )
        prediction_trainer = Trainer(model=current_model, args=pred_args)
        
        logger.info(f"Predicting on {len(unlabeled_processed_data_for_prediction)} unlabeled samples...")
        predictions_output = prediction_trainer.predict(unlabeled_processed_data_for_prediction)
        logits_unlabeled = predictions_output.predictions # Shape: (N, NumClasses, H, W)

        # --- 5.D. Select Samples for VLM Query based on Certainty ---
        num_to_evaluate_for_vlm = min(samples_to_evaluate_certainty_per_iter, len(unlabeled_indices))
        indices_to_consider_for_vlm_query_relative = random.sample(range(len(unlabeled_indices)), num_to_evaluate_for_vlm)
        
        certainty_scores = [] # List of (original_pool_idx, certainty_score, pred_mask_np_for_vlm, dominant_label_id_for_vlm)

        logger.info(f"Calculating certainty for {len(indices_to_consider_for_vlm_query_relative)} candidate samples...")
        for rel_idx in tqdm(indices_to_consider_for_vlm_query_relative, desc="Certainty Calc"):
            original_pool_idx = unlabeled_indices[rel_idx]
            sample_logits = torch.tensor(logits_unlabeled[rel_idx]).unsqueeze(0).to(device) # (1, NumClasses, H, W)
            
            # Upsample logits to original image size if necessary.
            # This is a critical step. Assuming Segformer output matches processed input size.
            # If preprocess_data resizes images to fixed size (e.g., 512x512), and model outputs logits at this size,
            # then direct use might be fine. Otherwise, upsampling to original image dimensions is needed.
            # For now, we pass sample_logits directly. The `calculate_prediction_certainty` needs to handle this.
            # The `pred_mask_np` should be derived from the same logits (potentially upsampled) used for certainty.
            
            # For Segformer, logits are typically 1/4 of input image resolution. They NEED upsampling.
            # Let's get the target size from the raw image associated with this sample.
            raw_img_pil = unlabeled_raw_data_for_vlm[rel_idx][dataset_config['image_col']]
            target_height, target_width = raw_img_pil.height, raw_img_pil.width

            upsampled_logits_for_certainty = torch.nn.functional.interpolate(
                sample_logits.float(), # Ensure float for interpolate
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            )

            pred_mask_tensor = torch.argmax(upsampled_logits_for_certainty.squeeze(0), dim=0) # (H_orig, W_orig)
            pred_mask_np = pred_mask_tensor.cpu().numpy().astype(np.uint8)

            unique_labels, counts = np.unique(pred_mask_np, return_counts=True)
            valid_labels_mask = unique_labels != 255 # Assuming 255 is ignore_index
            unique_labels = unique_labels[valid_labels_mask]
            counts = counts[valid_labels_mask]

            dominant_label_id_for_vlm = None
            if len(unique_labels) > 0:
                dominant_label_id_for_vlm = unique_labels[np.argmax(counts)]
            
            certainty = calculate_prediction_certainty(
                logits=upsampled_logits_for_certainty, 
                pred_mask_np=pred_mask_np,
                dominant_label_id=dominant_label_id_for_vlm,
                ignore_index=255
            )
            certainty_scores.append((original_pool_idx, certainty, pred_mask_np, dominant_label_id_for_vlm))

        certainty_scores.sort(key=lambda x: x[1]) # Sort by certainty (ascending, least certain first)
        num_to_query_vlm_this_iter = min(samples_to_query_vlm_per_iter, len(certainty_scores))
        selected_samples_for_vlm_query = certainty_scores[:num_to_query_vlm_this_iter]

        logger.info(f"Selected {len(selected_samples_for_vlm_query)} least certain samples for VLM query.")

        # --- 5.E. Query VLM for selected samples ---
        newly_pseudo_labeled_pool_indices = [] 
        vlm_agreements = 0

        for original_pool_idx, _, pred_mask_np_for_vlm, dominant_label_id_for_vlm in tqdm(selected_samples_for_vlm_query, desc="VLM Querying"):
            # Find relative index in unlabeled_indices to fetch the correct raw image
            try:
                rel_idx_in_unlabeled = unlabeled_indices.index(original_pool_idx)
            except ValueError: # Should not happen if logic is correct
                logger.error(f"Error: original_pool_idx {original_pool_idx} not found in unlabeled_indices during VLM query phase.")
                continue

            image_for_vlm = unlabeled_raw_data_for_vlm[rel_idx_in_unlabeled][dataset_config['image_col']] # PIL Image
            pred_mask_pil_for_vlm = Image.fromarray(pred_mask_np_for_vlm, mode='L') # From certainty calculation

            if dominant_label_id_for_vlm is not None and dominant_label_id_for_vlm != 255:
                dominant_label_name = PASCAL_VOC_ID2LABEL.get(dominant_label_id_for_vlm, "unknown object")
                question = vlm_query_template.format(label_name=dominant_label_name)
                
                try:
                    vlm_response = vlm_handler.get_vlm_feedback_for_sample(
                        image=image_for_vlm,
                        predicted_mask=pred_mask_pil_for_vlm,
                        dominant_class_id=int(dominant_label_id_for_vlm), # Ensure it's int
                        question=question,
                    )

                    if vlm_response.lower().strip() == "yes":
                        pseudo_labels_for_indices[original_pool_idx] = pred_mask_pil_for_vlm
                        newly_pseudo_labeled_pool_indices.append(original_pool_idx)
                        vlm_agreements +=1
                except Exception as e:
                    logger.error(f"Error during VLM query for sample (pool idx {original_pool_idx}): {e}", exc_info=True)
            else:
                logger.debug(f"Skipping VLM query for sample (pool idx {original_pool_idx}) as no valid dominant class found.")
        
        if general_config.get('wandb_enabled', False) and wandb:
            wandb.log({
                f"iter_{iteration+1}_vlm/pseudo_labels_added": len(newly_pseudo_labeled_pool_indices),
                f"iter_{iteration+1}_vlm/agreements": vlm_agreements,
                f"iter_{iteration+1}_vlm/queries_made": len(selected_samples_for_vlm_query) # Number of samples sent to VLM
            }, step=iteration+1)


        # --- 5.F. Update Datasets for Next Iteration ---
        if newly_pseudo_labeled_pool_indices:
            # These are now considered "labeled" for the next round by virtue of VLM agreement.
            # They are already in pseudo_labels_for_indices.
            # We need to move them from unlabeled_indices to current_gt_labeled_indices for the *next* iteration's GT pool.
            
            # Add to GT pool for next iteration
            current_gt_labeled_indices.extend(newly_pseudo_labeled_pool_indices)
            current_gt_labeled_indices = sorted(list(set(current_gt_labeled_indices)))
            
            # Remove from unlabeled pool
            unlabeled_indices = sorted(list(set(unlabeled_indices) - set(newly_pseudo_labeled_pool_indices)))
            
            # The pseudo-labels in pseudo_labels_for_indices for these newly_pseudo_labeled_pool_indices
            # will be consumed in the next iteration's data prep. After that, they are effectively part of GT.
            # So, we can clear them from pseudo_labels_for_indices *after* they are used to form the next training set,
            # or simply let them be overwritten if re-queried (though they shouldn't be if moved out of unlabeled_indices).
            # For simplicity, items in newly_pseudo_labeled_pool_indices are now "GT-like" for the next iter.
            # The pseudo_labels_for_indices store will be used to build the pseudo part of the *next* training set.
            # If an index is in current_gt_labeled_indices, it will be sourced from processed_train_pool.
            # If an index is in pseudo_labels_for_indices, it will be built from raw + PIL mask.
            # The current logic correctly separates these.
            # We *do not* remove from pseudo_labels_for_indices here, as they are needed for the *next* iteration's pseudo data construction.
            # They are removed from `unlabeled_indices` so they are not candidates for VLM query again.
            # And they are added to `current_gt_labeled_indices` so they are part of the "known" set.
            # This means in the next iteration, if an index is in BOTH `current_gt_labeled_indices` AND `pseudo_labels_for_indices`,
            # the GT version from `processed_train_pool` will be used. This is fine.
            # The `pseudo_labels_for_indices` should ideally only contain labels for things *not yet* in `current_gt_labeled_indices`.
            # Let's refine: once a pseudo-label is accepted and the index moves to `current_gt_labeled_indices`,
            # it should no longer be in `pseudo_labels_for_indices` because its "ground truth" is now the VLM-confirmed mask,
            # which will be part of the GT dataset construction.
            
            # Refined logic:
            # The samples in newly_pseudo_labeled_pool_indices have their pseudo-labels stored in pseudo_labels_for_indices.
            # These will be used in the *next* iteration to construct the "pseudo" part of the training data.
            # After that training, they effectively become part of the "known" set.
            # The current_gt_labeled_indices are those with original ground truth.
            # This seems okay. The key is that `unlabeled_indices` shrinks.

        logger.info(f"End of VLM Iteration {iteration + 1}:")
        logger.info(f"  GT Labeled samples (original GT + VLM confirmed for next iter): {len(current_gt_labeled_indices)}")
        logger.info(f"  Unlabeled samples remaining for query: {len(unlabeled_indices)}")
        logger.info(f"  Current pseudo-label store size (for next iter pseudo data): {len(pseudo_labels_for_indices)}")
        if general_config.get('wandb_enabled', False) and wandb:
            wandb.log({
                f"iter_{iteration+1}_dataset/total_gt_labeled_for_next_iter": len(current_gt_labeled_indices),
                f"iter_{iteration+1}_dataset/total_unlabeled_for_query": len(unlabeled_indices),
                f"iter_{iteration+1}_dataset/pseudo_label_store_size": len(pseudo_labels_for_indices),
            }, step=iteration+1)

    # --- 6. Final Evaluation on Test Set ---
    if current_model and processed_test_dataset and len(processed_test_dataset) > 0:
        logger.info("\n--- Final Evaluation on FIXED Test Set ---")
        final_eval_output_dir = os.path.join(output_dir, "final_evaluation")
        os.makedirs(final_eval_output_dir, exist_ok=True)

        final_eval_args = TrainingArguments(
            output_dir=final_eval_output_dir,
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            fp16=training_config.get('fp16', False) and torch.cuda.is_available(),
            report_to="none",
            remove_unused_columns=False,
        )
        final_trainer = Trainer(
            model=current_model,
            args=final_eval_args,
            eval_dataset=processed_test_dataset,
            compute_metrics=compute_metrics_fn, # Defined earlier
        )
        test_metrics = final_trainer.evaluate(eval_dataset=processed_test_dataset)
        logger.info(f"Final Test Set Metrics: {test_metrics}")

        final_model_path = os.path.join(output_dir, "final_model_from_vlm_itl")
        current_model.save_pretrained(final_model_path)
        
        test_metrics_path = os.path.join(final_eval_output_dir, "test_results.json")
        with open(test_metrics_path, "w") as f:
            json.dump(test_metrics, f, indent=4)
        logger.info(f"Final model saved to {final_model_path}")
        logger.info(f"Final test metrics saved to {test_metrics_path}")

        if general_config.get('wandb_enabled', False) and wandb:
            wandb.log({"final_test/mean_iou": test_metrics.get('eval_mean_iou', 0.0), # Log specific key metrics
                       "final_test/accuracy": test_metrics.get('eval_accuracy', 0.0)}) 
            # Log all test metrics
            for k, v in test_metrics.items():
                wandb.summary[f"final_test_{k.replace('eval_','')}"] = v


    elif not current_model:
         logger.error("VLM-ITL loop did not produce a final model. Skipping final evaluation.")
    else: 
         logger.warning("No test set available or processed test set is empty. Skipping final evaluation.")

    if general_config.get('wandb_enabled', False) and wandb and wandb.run:
        wandb.finish()
    logger.info("VLM-ITL pipeline finished.")


if __name__ == "__main__":
    # Argument parsing should be here, similar to train_active_learning_overlap.py
    # For now, assume config path is hardcoded or passed directly for testing.
    # Example:
    # config_file_path = 'configs/vlm_itl_config.yaml' 
    # run_vlm_itl_pipeline(config_file_path)

    # Using argparse like in the other script:
    import argparse # Make sure argparse is imported
    parser = argparse.ArgumentParser(description="Run VLM-In-The-Loop pipeline for Image Segmentation")
    parser.add_argument(
        "--config",
        type=str,
        default='configs/vlm_itl_config.yaml', # Default config path
        help="Path to the VLM-ITL configuration YAML file."
    )
    args = parser.parse_args()

    # Setup basic logging if not already configured by logger setup
    logging.basicConfig(
        level=logging.INFO, # Changed to INFO to see more details by default
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    run_vlm_itl_pipeline(args.config)