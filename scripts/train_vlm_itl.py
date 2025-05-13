from dotenv import load_dotenv
load_dotenv()
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
from utils.vlm import get_vlm_handler, HuggingFaceVLMHandler # Assuming this is where VLM handlers are defined
from utils.active_learning import ( # For SegmentationImageLoggerCallbackVLM if used
    # sample_initial_data, 
    # select_next_batch_indices,
    # feature_extractor_fn,
    # ActiveLearningProgressCallback, 
    SegmentationImageLoggerCallback, # Or a VLM specific version
    compute_image_uncertainties,
    mock_compute_image_uncertainties
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
    PASCAL_VOC_IGNORE_INDEX
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
    # set HF_HOME env variable if not set
    hf_home = os.getenv("HF_HOME")
    logging.info(f"HF_HOME: {hf_home}")

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
        vlm_handler_config = vlm_config.get('vlm_handler', {})
        vlm_handler = HuggingFaceVLMHandler(vlm_handler_config)
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
    full_dataset = raw_datasets['train'].map(
        shared_preprocess_fn,
        batched=True,
        batch_size=training_config.get('per_device_eval_batch_size', 8),
        remove_columns=[dataset_config['image_col'], dataset_config['mask_col']],
        load_from_cache_file=general_config.get('load_from_cache_file', True),
        desc="Preprocessing raw training data"
    )
    full_dataset.set_format("torch", columns=["pixel_values", "labels"])
    
    full_dataset_dict = DatasetDict({
        'train': full_dataset.select(train_indices),
        'validation': full_dataset.select(val_indices),
        'test': full_dataset.select(test_indices)
    })

    processed_train_pool = full_dataset.select(train_indices)
    processed_val_dataset = full_dataset.select(val_indices)
    processed_test_dataset = full_dataset.select(test_indices)
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
    

    logger.info(f"Loading segmentation model")
    current_model = load_model_for_segmentation(
        model_name_or_path=model_config['name'],
        num_labels=NUM_PASCAL_VOC_LABELS,
        id2label=PASCAL_VOC_ID2LABEL,
        label2id=PASCAL_VOC_LABEL2ID,
        ignore_mismatched_sizes=model_config.get('ignore_mismatched_sizes', True)
    )
    current_model = None
    
    current_training_indices = current_gt_labeled_indices.copy()
    for iteration in range(num_vlm_iterations):
        logger.info(f"--- VLM Iteration {iteration + 1} / {num_vlm_iterations} ---")
        
        iter_output_dir = os.path.join(output_dir, f"iteration_{iteration + 1}")
        os.makedirs(iter_output_dir, exist_ok=True)
        
        # --- 5.A. Prepare Training Data for Current Iteration ---
        if unlabeled_indices and current_model is not None:
            k = 100
            if general_config["active_learning"]["sampling_strategy"] == "mock":
                uncertainties, segmentations = mock_compute_image_uncertainties(
                    model=current_model,
                    dataset=processed_train_pool,
                    remaining_indices=unlabeled_indices,
                    preprocess_fn=shared_preprocess_fn,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    batch_size=training_config.get("per_device_eval_batch_size", 8),
                )
            else:
                uncertainties, segmentations = compute_image_uncertainties(
                    model=current_model,
                    dataset=processed_train_pool,
                    remaining_indices=unlabeled_indices,
                    preprocess_fn=shared_preprocess_fn,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    batch_size=training_config.get("per_device_eval_batch_size", 8),
                )
            logger.info(f"Uncertainties calculated for {len(unlabeled_indices)} unlabeled samples.")
            # Pick top-k by descending entropy
            new_indicies = sorted(
                uncertainties, 
                key=lambda idx: uncertainties[idx], 
                reverse=True,
                # sort by ascending entropy instead for min_entropy
                # reverse=True, 
            )[:k]

            # # Modify segmentations for new_indicies to keep only the dominant class
            # for idx_to_modify in new_indicies:
                    
            #     current_mask_np = segmentations[idx_to_modify]
            #     unique_classes, counts = np.unique(current_mask_np, return_counts=True)
                
            #     dominant_class_id = -1
            #     max_count = -1
            #     logger.info(f"Unique classes in mask {idx_to_modify}: {unique_classes}, counts: {counts}")
            #     # Find the dominant foreground class
            #     for class_id, count in zip(unique_classes, counts):
            #         if class_id == PASCAL_VOC_IGNORE_INDEX or class_id == 0: # Ignore index or background
            #             continue
            #         if count > max_count:
            #             max_count = count
            #             dominant_class_id = class_id
                
            #     # Create new mask: fill with background (0), then add only dominant class pixels
            #     new_mask_np = np.full(current_mask_np.shape, 0, dtype=current_mask_np.dtype) 
            #     if dominant_class_id != -1: # If a dominant foreground class was found
            #         new_mask_np[current_mask_np == dominant_class_id] = dominant_class_id

            #     segmentations[idx_to_modify] = new_mask_np
            # logger.info(f"Processed {len(new_indicies)} masks to keep only dominant foreground class.")

            good_indices = []
            bad_indices  = []

            for idx in new_indicies:
                # Get the original image and segmentation mask as PIL images
                original_image = raw_train_data_subset[idx][dataset_config['image_col']]
                segmentation_mask = Image.fromarray(segmentations[idx].astype(np.uint8))
                vlm_feed_back = vlm_handler.ask_binary_question(
                    image = original_image,
                    segmentation_mask = segmentation_mask,
                    prompt=vlm_config["vlm_query_template"],
                    idx=f"{iteration}_{idx}",
                )

                if vlm_feed_back:
                    logger.info(f"Dominant class ID: {np.unique(segmentations[idx])}, VLM feedback: {vlm_feed_back}, {idx}")
                    good_indices.append(idx)
                else:
                    bad_indices.append(idx)
            logger.info(f"{len(good_indices)}/{len(new_indicies)} passed VLM check")

            def _override_mask(example, example_idx):
                if example_idx in good_indices:
                    predicted_mask_np = segmentations[example_idx]
                    
                    # Convert to PIL Image
                    pil_mask = Image.fromarray(predicted_mask_np.astype(np.uint8))
                    
                    # Resize to the target size used by the image_processor for labels
                    # image_processor.size is a dict like {'height': H, 'width': W}
                    # PIL resize takes (width, height)
                    resized_pil_mask = pil_mask.resize(
                        (image_processor.size["width"], image_processor.size["height"]),
                        resample=Image.NEAREST # Use NEAREST for segmentation masks
                    )
                    resized_mask_np = np.array(resized_pil_mask)
                    
                    return {"labels": resized_mask_np.tolist()}
                else:
                    # return no change
                    return {}

                # this will go over your entire train split, but only rewrite the
            # 'mask' for good_indices
            processed_train_pool = processed_train_pool.map(
                _override_mask,
                with_indices=True,
            )

            current_training_indices.extend(good_indices)
            unlabeled_indices = sorted(list(set(unlabeled_indices) - set(good_indices)))
            logger.info(f"Added {len(good_indices)} auto-labeled samples, kept {len(bad_indices)} still unlabeled.")
            
        
        current_training_subset = processed_train_pool.select(current_training_indices)
        if current_model is None: # First iteration
            logger.info("Loading initial model...")
            current_model = load_model_for_segmentation(
                model_name_or_path=config['model']['name'],
                num_labels=NUM_PASCAL_VOC_LABELS,
                id2label=PASCAL_VOC_ID2LABEL,
                label2id=PASCAL_VOC_LABEL2ID,
                ignore_mismatched_sizes=config['model'].get('ignore_mismatched_sizes', False)
            )
        current_model.to(device)

        iter_training_args_output_dir = os.path.join(iter_output_dir, "training_checkpoints")
        os.makedirs(iter_training_args_output_dir, exist_ok=True)

        iter_training_args = TrainingArguments(
            output_dir=iter_training_args_output_dir,
            num_train_epochs=training_config.get('num_train_epochs', 3),
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            save_strategy=training_config.get("save_strategy_per_iter", "epoch"),
            eval_strategy=training_config.get("evaluation_strategy_per_iter", "epoch"),
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
            model=current_model,
            args=iter_training_args,
            train_dataset=current_training_subset,
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

        logger.info("-"*50)
        logger.info(f"End of VLM Iteration {iteration + 1}:")
        logger.info(f"  Original GT Labeled samples count: {len(current_gt_labeled_indices)}")
        logger.info(f"  VLM-confirmed pseudo-labeled samples count: {len(current_training_indices)}")
        logger.info(f"  Unlabeled samples remaining for query: {len(unlabeled_indices)}")
        if general_config.get('wandb_enabled', False) and wandb:
            wandb.log({
                f"iter_{iteration+1}_dataset/original_gt_sample_count": len(current_gt_labeled_indices),
                f"iter_{iteration+1}_dataset/vlm_confirmed_pseudo_sample_count": len(pseudo_labels_for_indices),
                f"iter_{iteration+1}_dataset/unlabeled_sample_for_query_count": len(unlabeled_indices),
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