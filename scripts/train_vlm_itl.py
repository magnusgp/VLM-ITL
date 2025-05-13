import os
import sys
import math
import random
import numpy as np
import torch
from functools import partial
from tqdm import tqdm
from PIL import Image
from typing import List, Dict, Any, Tuple, Callable, Optional, Iterator

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from transformers import (
    TrainingArguments,
    Trainer,
    SegformerImageProcessor,
    set_seed,
    EarlyStoppingCallback
)
# Use HFDataset for the new approach
from datasets import Dataset as HFDataset, Features
from datasets import Image as HFImageField # For specifying features

from utils.config import load_config
from utils.log_utils import setup_wandb, logger
from utils.metrics import compute_metrics_segmentation
from utils.vlm import get_vlm_handler, VLMHandler
from data.pascal_voc import (
    load_pascal_voc_dataset,
    preprocess_data,
    create_train_val_test_splits,
    PASCAL_VOC_ID2LABEL,
    PASCAL_VOC_LABEL2ID,
    NUM_PASCAL_VOC_LABELS
)
from models.segformer import load_model_for_segmentation

# Add a custom Trainer class for explicit loss handling
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
            labels = labels.squeeze(1)
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

# Helper function to convert various mask types to numpy arrays
def ensure_mask_as_numpy(mask_data) -> np.ndarray:
    """
    Convert mask data to a numpy array regardless of input type.
    
    Args:
        mask_data: Can be a PIL Image, numpy array, or path to an image
        
    Returns:
        np.ndarray: The mask as a numpy array
    """
    if isinstance(mask_data, np.ndarray):
        return mask_data
    elif isinstance(mask_data, Image.Image):
        return np.array(mask_data)
    elif isinstance(mask_data, dict) and 'path' in mask_data:
        return np.array(Image.open(mask_data['path']).convert('L'))
    elif isinstance(mask_data, str):
        return np.array(Image.open(mask_data).convert('L'))
    else:
        try:
            return np.array(mask_data)
        except Exception as e:
            logger.error(f"Cannot convert mask to numpy: {e}")
            return np.zeros((10, 10), dtype=np.uint8)  # Fallback empty mask

def get_dominant_label_and_mask_from_prediction(
    pred_mask_np: np.ndarray,
    id2label: Dict[int, str],
    ignore_labels: List[int] = [0, 255] # Background and ignore index for PASCAL
) -> Tuple[Optional[str], Optional[Image.Image], Optional[int]]:
    """
    Finds the dominant predicted label name, its corresponding binary mask, and its ID.
    """
    unique_labels, counts = np.unique(pred_mask_np, return_counts=True)
    dominant_label_id = -1
    max_count = -1

    for label_id, count in zip(unique_labels, counts):
        if label_id in ignore_labels:
            continue
        if count > max_count:
            max_count = count
            dominant_label_id = int(label_id) # Ensure it's Python int

    if dominant_label_id != -1 and dominant_label_id in id2label :
        dominant_label_name = id2label.get(dominant_label_id, "unknown")
        # Create a binary mask for the dominant label (mask values are 0 or dominant_label_id)
        # For VLM verification, we often care about the region of this dominant class.
        # The pseudo-label saved should be the full multi-class pred_mask_np if VLM confirms the dominant part.
        # Or, if VLM confirms "object X is there and segmented well", we save pred_mask_np.
        # Let's save the original multi-class predicted mask if VLM confirms its dominant aspect.
        
        # For the purpose of asking VLM, we might want to give it a binary mask of the dominant object
        # dominant_object_mask_np = (pred_mask_np == dominant_label_id).astype(np.uint8) * 255 # Binary 0 or 255
        # dominant_object_mask_pil = Image.fromarray(dominant_object_mask_np, mode='L')
        # For now, let's assume the VLM can be queried on the whole image + dominant label name,
        # and the `segmentation_mask` argument to `ask_binary_question` can be the `pred_mask_pil` (multi-class).

        return dominant_label_name, Image.fromarray(pred_mask_np.astype(np.uint8), mode='L'), dominant_label_id
    return None, None, None


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
    if dominant_label_id is None or dominant_label_id == -1:
        return 0.0

    probs = torch.softmax(logits.squeeze(0), dim=0)  # (NumClasses, H, W)
    
    # Probabilities for the dominant class
    dominant_class_probs = probs[dominant_label_id, :, :] # (H, W)
    
    # Mask for pixels belonging to the dominant class in the prediction
    dominant_class_pixel_mask = torch.from_numpy(pred_mask_np == dominant_label_id).to(dominant_class_probs.device)
    
    if dominant_class_pixel_mask.sum() == 0:
        return 0.0 # No pixels predicted as the dominant class
        
    # Average probability for the dominant class over its predicted pixels
    certainty = torch.sum(dominant_class_probs * dominant_class_pixel_mask) / torch.sum(dominant_class_pixel_mask)
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
    run_name = f"{vlm_config.get('run_name_prefix', 'vlm_itl')}_{general_config['seed']}_{random.randint(1000,9999)}"
    output_dir = os.path.join(output_dir_base, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Run output will be saved to: {output_dir}")

    if general_config.get('wandb_enabled', False):
        setup_wandb(config, run_name, project_suffix="-vlm_itl")

    # --- 2. Initialize VLM Handler (BLIP) ---
    logger.info("Initializing VLM Handler (HuggingFace BLIP)...")
    try:
        vlm_handler_config = config.copy() # Use a copy to modify for VLM handler init
        vlm_handler_config['vlm_itl']['vlm_handler'] = 'huggingface_blip' 
        # Ensure vlm_options for model name if not default, e.g.
        # vlm_handler_config['vlm_itl'].setdefault('vlm_options', {})['vlm_model_name'] = 'Salesforce/blip-vqa-base'
        vlm_handler = get_vlm_handler(vlm_handler_config)
    except Exception as e:
        logger.error(f"Failed to initialize VLM Handler: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Load and Prepare Full Data ---
    logger.info(f"Loading PASCAL VOC dataset: {dataset_config['name']}...")
    # This should load the full dataset, including images that might be paths
    raw_datasets = load_pascal_voc_dataset(
        dataset_name=dataset_config['name'],
        cache_dir=dataset_config.get('cache_dir') 
    )
    
    image_processor = SegformerImageProcessor.from_pretrained(
        dataset_config['feature_extractor_name'],
        do_reduce_labels=False,
        do_resize=True # Explicitly set to False to let preprocess_data handle label mapping
    )
    # image_processor.do_normalize = model_config.get('do_normalize', True)
    # image_processor.image_mean = model_config.get('image_mean', [0.485, 0.456, 0.406])
    # image_processor.image_std = model_config.get('image_std', [0.229, 0.224, 0.225])

    logger.info("Creating initial data splits...")
    full_train_data: HFDataset = raw_datasets['train'] # This is an ArrowDataset
    val_dataset: HFDataset = raw_datasets['test'] # Or derive from train if no val split
    test_dataset: HFDataset = raw_datasets['test'] # Or derive

    # If val/test are not pre-defined, split them from full_train_data
    if 'validation' not in raw_datasets or 'test' not in raw_datasets:
        logger.info("Validation/Test splits not found in raw_datasets. Creating from training data.")
        full_train_data, val_dataset, test_dataset = create_train_val_test_splits(
            raw_datasets['train'], 
            val_percentage=dataset_config.get('val_split_percentage', 0.1),
            test_percentage=dataset_config.get('test_split_percentage', 0.1),
            seed=general_config['seed']
        )
    
    logger.info(f"Full train data size: {len(full_train_data)}")
    logger.info(f"Validation data size: {len(val_dataset)}")
    logger.info(f"Test data size: {len(test_dataset)}")

    initial_train_percentage = vlm_config.get('initial_training_percentage', 0.05) # e.g. 5%
    num_initial_samples = math.ceil(initial_train_percentage * len(full_train_data))
    
    all_train_indices = list(range(len(full_train_data)))
    random.shuffle(all_train_indices) 
    
    # These are indices with respect to `full_train_data`
    current_gt_labeled_indices = sorted(all_train_indices[:num_initial_samples])
    unlabeled_indices = sorted(all_train_indices[num_initial_samples:])
    
    pseudo_labels_for_indices: Dict[int, Image.Image] = {} 

    logger.info(f"Initial GT labeled set size: {len(current_gt_labeled_indices)}")
    logger.info(f"Initial unlabeled pool size: {len(unlabeled_indices)}")

    # --- 4. Preprocessing Function ---
    shared_preprocess_fn = partial(
        preprocess_data, # Your existing function from pascal_voc.py
        image_processor=image_processor,
        image_col=dataset_config['image_col'],
        mask_col=dataset_config['mask_col']
    )
    
    # Preprocess val and test datasets once (they don't change)
    logger.info("Preprocessing validation dataset...")
    processed_val_dataset = val_dataset.map(
        shared_preprocess_fn, 
        batched=True, 
        remove_columns=val_dataset.column_names,
        batch_size=training_config.get('per_device_eval_batch_size', 8),
        load_from_cache_file=False # Force reprocessing if needed
    )
    processed_val_dataset.set_format("torch", columns=["pixel_values", "labels"])
    logger.info("Preprocessing test dataset...")
    processed_test_dataset = test_dataset.map(
        shared_preprocess_fn, 
        batched=True, 
        remove_columns=test_dataset.column_names,
        batch_size=training_config.get('per_device_eval_batch_size', 8),
        load_from_cache_file=False # Force reprocessing if needed
    )
    processed_test_dataset.set_format("torch")
    
    # Preprocess the full training data once too - this is important for consistent format
    logger.info("Preprocessing full training dataset (will be used for predictions)...")
    processed_full_train_data = full_train_data.map(
        shared_preprocess_fn,
        batched=True,
        remove_columns=full_train_data.column_names,
        batch_size=training_config.get('per_device_train_batch_size', 8),
        load_from_cache_file=False # Force reprocessing if needed
    )
    processed_full_train_data.set_format("torch")
    logger.info(f"Processed full training data size: {len(processed_full_train_data)}")


    # --- 5. VLM Iteration Loop ---
    num_vlm_iterations = vlm_config.get('num_vlm_iterations', 5)
    samples_to_evaluate_certainty_per_iter = vlm_config.get('samples_to_evaluate_certainty_per_iter', 200)
    samples_to_query_vlm_per_iter = vlm_config.get('samples_to_query_vlm_per_iter', 50)
    vlm_query_template = vlm_config.get('vlm_query_template', "Is the primary object in this segmented region a {label_name}?")


    for iteration in range(num_vlm_iterations):
        logger.info(f"--- VLM Iteration {iteration + 1} / {num_vlm_iterations} ---")
        
        # --- 5.A. Prepare Training Data for Current Iteration ---
        logger.info("Preparing training data for current iteration...")
        
        # Split the data into ground truth indices and pseudo-labeled indices
        gt_indices = current_gt_labeled_indices.copy()
        pseudo_indices = list(pseudo_labels_for_indices.keys())
        
        if not gt_indices and not pseudo_indices:
            logger.warning("No training data (GT or pseudo) available. Skipping training for this iteration.")
            if iteration == 0:
                logger.error("No initial training data. Check initial_training_percentage or data loading.")
                break
            continue
            
        logger.info(f"Training with {len(gt_indices)} ground truth samples and {len(pseudo_indices)} pseudo-labeled samples")
        
        # Create a new dataset for this iteration that combines ground truth and pseudo-labeled data
        # 1. Get the original samples for ground truth data
        if gt_indices:
            gt_dataset_raw = full_train_data.select(gt_indices)
        else:
            gt_dataset_raw = None
            
        # 2. For pseudo-labeled samples, we need to replace the masks
        if pseudo_indices:
            # Start by selecting the samples with their original images
            pseudo_dataset_raw = full_train_data.select(pseudo_indices)
            
            # Create a new dataset with the updated masks
            pseudo_samples = []
            for i, idx in enumerate(pseudo_indices):
                # Get original sample for the image
                original_sample = full_train_data[idx]
                
                # Ensure pseudo mask is in numpy format for HF Dataset
                pseudo_mask_np = ensure_mask_as_numpy(pseudo_labels_for_indices[idx])
                
                # Create new sample with original image but pseudo mask
                new_sample = {
                    dataset_config['image_col']: original_sample[dataset_config['image_col']],
                    dataset_config['mask_col']: pseudo_mask_np
                }
                pseudo_samples.append(new_sample)
                
            # Create dataset from the prepared samples
            from datasets import Dataset as HFDataset
            pseudo_dataset_raw = HFDataset.from_dict({
                dataset_config['image_col']: [s[dataset_config['image_col']] for s in pseudo_samples],
                dataset_config['mask_col']: [s[dataset_config['mask_col']] for s in pseudo_samples]
            })
        else:
            pseudo_dataset_raw = None
            
        # 3. Combine the raw datasets
        if gt_dataset_raw is not None and pseudo_dataset_raw is not None:
            from datasets import concatenate_datasets
            combined_dataset_raw = concatenate_datasets([gt_dataset_raw, pseudo_dataset_raw])
        elif gt_dataset_raw is not None:
            combined_dataset_raw = gt_dataset_raw
        else:
            combined_dataset_raw = pseudo_dataset_raw
            
        # 4. Apply preprocessing to the combined dataset
        logger.info(f"Preprocessing combined dataset with {len(combined_dataset_raw)} samples...")
        processed_iter_train_dataset = combined_dataset_raw.map(
            shared_preprocess_fn,
            batched=True,
            batch_size=training_config.get('per_device_train_batch_size', 8),
            remove_columns=combined_dataset_raw.column_names,
            load_from_cache_file=False # Force reprocessing to avoid cache issues
        )
        
        # 5. Set the torch format for the processed dataset
        processed_iter_train_dataset.set_format("torch", columns=["pixel_values", "labels"])
        logger.info(f"Final processed training dataset for iteration {iteration+1}: {len(processed_iter_train_dataset)} samples")

        # --- 5.B. Train Segmentation Model ---
        model = load_model_for_segmentation(
            model_name_or_path=model_config['name'],
            num_labels=NUM_PASCAL_VOC_LABELS,
            id2label=PASCAL_VOC_ID2LABEL,
            label2id=PASCAL_VOC_LABEL2ID,
            ignore_mismatched_sizes=model_config.get('ignore_mismatched_sizes', True)
        )
        model = model.to(device)  # Ensure model is explicitly moved to device

        # Adjust epochs per VLM iteration if needed
        num_train_epochs_this_iter = training_config.get('num_epochs_per_vlm_iter', training_config.get('num_epochs', 3))

        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, f"training_iter_{iteration}"),
            num_train_epochs=num_train_epochs_this_iter,
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            learning_rate=float(training_config['learning_rate']),
            weight_decay=training_config['weight_decay'],
            eval_strategy="epoch" if num_train_epochs_this_iter > 0 else "no",
            save_strategy="epoch" if num_train_epochs_this_iter > 0 else "no",
            load_best_model_at_end=True if num_train_epochs_this_iter > 0 else False,
            metric_for_best_model="eval_mean_iou",
            greater_is_better=True,
            logging_dir=os.path.join(output_dir, f"logs_iter_{iteration}"),
            logging_steps=training_config.get('logging_steps', 50),
            remove_unused_columns=False,
            report_to="wandb" if general_config.get('wandb_enabled', False) else "none",
        )

        trainer = SegmentationTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_iter_train_dataset if len(processed_iter_train_dataset) > 0 else None,
            eval_dataset=processed_val_dataset,
            compute_metrics=compute_metrics_segmentation,
        )
        
        if len(processed_iter_train_dataset) > 0 and num_train_epochs_this_iter > 0:
            logger.info(f"Starting model training for iteration {iteration + 1} ({num_train_epochs_this_iter} epochs)...")
            train_result = trainer.train()
            logger.info(f"Training finished. Metrics: {train_result.metrics}")
            model = trainer.model # Use the best model loaded by trainer
        elif len(processed_iter_train_dataset) == 0 :
             logger.info("Skipping training as training dataset is empty.")
        else: # num_train_epochs_this_iter is 0
             logger.info("Skipping training as num_epochs_per_vlm_iter is 0.")


        # --- 5.C. Predict on Unlabeled Data & Select Most Certain Samples ---
        if not unlabeled_indices:
            logger.info("No unlabeled data left. Ending VLM iterations.")
            break
        
        num_to_eval_cert = min(len(unlabeled_indices), samples_to_evaluate_certainty_per_iter)
        indices_to_eval_certainty_this_iter = random.sample(unlabeled_indices, num_to_eval_cert)
        logger.info(f"Predicting on {len(indices_to_eval_certainty_this_iter)} unlabeled samples to find candidates for VLM...")
        
        model.eval()
        candidate_predictions = [] # List of (certainty, original_idx, image_pil, pred_mask_pil_full, dominant_label_name)
        
        with torch.no_grad():
            for original_idx in tqdm(indices_to_eval_certainty_this_iter, desc="Certainty Calculation"):
                # Get preprocessed tensor data for this sample
                try:
                    preprocessed_sample = processed_full_train_data[original_idx]
                    pixel_values = preprocessed_sample["pixel_values"].unsqueeze(0).to(device)  # Add batch dimension
                except Exception as e:
                    # If there's an issue with the preprocessed data, process on-the-fly
                    logger.warning(f"Error using preprocessed data for prediction on sample {original_idx}: {e}. Processing on-the-fly.")
                    raw_sample = full_train_data[original_idx]
                    image_pil_val = raw_sample[dataset_config['image_col']]
                    if isinstance(image_pil_val, Image.Image):
                        image_pil = image_pil_val
                    elif isinstance(image_pil_val, dict) and 'path' in image_pil_val:
                        image_pil = Image.open(image_pil_val['path']).convert("RGB")
                    elif isinstance(image_pil_val, str): # Direct path string
                        image_pil = Image.open(image_pil_val).convert("RGB")
                    else:
                        logger.warning(f"Unexpected image format for sample {original_idx}. Skipping.")
                        continue
                    
                    # Process the image directly with the image processor
                    inputs = image_processor(images=image_pil, return_tensors="pt")
                    pixel_values = inputs.pixel_values.to(device)
                
                # For VLM verification later, we need the original image
                raw_sample = full_train_data[original_idx]
                image_pil_val = raw_sample[dataset_config['image_col']]
                if isinstance(image_pil_val, Image.Image):
                    image_pil = image_pil_val
                elif isinstance(image_pil_val, dict) and 'path' in image_pil_val:
                    image_pil = Image.open(image_pil_val['path']).convert("RGB")
                elif isinstance(image_pil_val, str): # Direct path string
                    image_pil = Image.open(image_pil_val).convert("RGB")
                else:
                    logger.warning(f"Unexpected image format for sample {original_idx}. Skipping.")
                    continue

                # Run model on preprocessed tensors
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits 

                original_height, original_width = image_pil.size[1], image_pil.size[0]
                upsampled_logits = torch.nn.functional.interpolate(
                    logits, size=(original_height, original_width), mode="bilinear", align_corners=False
                )
                pred_mask_tensor = upsampled_logits.argmax(dim=1) 
                pred_mask_np = pred_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
                
                # Get dominant label and the full predicted mask (as PIL)
                dominant_label_name, pred_mask_pil_full, dominant_label_id = get_dominant_label_and_mask_from_prediction(
                    pred_mask_np, PASCAL_VOC_ID2LABEL
                )

                if dominant_label_name and pred_mask_pil_full:
                    certainty = calculate_prediction_certainty(upsampled_logits.cpu(), pred_mask_np, dominant_label_id)
                    candidate_predictions.append(
                        (certainty, original_idx, image_pil, pred_mask_pil_full, dominant_label_name)
                    )
        
        candidate_predictions.sort(key=lambda x: x[0], reverse=True) # Sort by certainty
        vlm_query_candidates = candidate_predictions[:samples_to_query_vlm_per_iter]
        logger.info(f"Selected {len(vlm_query_candidates)} most certain candidates for VLM verification.")

        # --- 5.D. VLM Verification ---
        newly_verified_indices_count = 0
        if not vlm_query_candidates:
            logger.info("No candidates to send to VLM this iteration.")
        
        for cert, orig_idx, img_pil, pred_mask_pil_to_save, dom_label_name in tqdm(vlm_query_candidates, desc="VLM Verification"):
            if orig_idx not in unlabeled_indices: continue # Already processed

            vlm_query = vlm_query_template.format(label_name=dom_label_name)
            try:
                # Pass the full predicted mask (multi-class) as context if VLM can use it.
                # The question is about the dominant label found in that mask.
                vlm_confirms = vlm_handler.ask_binary_question(
                    image=img_pil,
                    segmentation_mask=pred_mask_pil_to_save, 
                    prompt=vlm_query
                )
            except Exception as e:
                logger.error(f"Error during VLM query for sample {orig_idx} (label: {dom_label_name}): {e}. Skipping.", exc_info=True)
                vlm_confirms = False

            if vlm_confirms:
                # Convert the predicted mask to numpy array and store it
                pseudo_labels_for_indices[orig_idx] = ensure_mask_as_numpy(pred_mask_pil_to_save)
                newly_verified_indices_count += 1
        
        if newly_verified_indices_count > 0:
            # Update unlabeled_indices by removing ALL keys from pseudo_labels_for_indices
            # This ensures if a sample was previously pseudo-labeled and re-queried (not current logic, but for future), it's handled.
            # More simply for now: remove those that just got added.
            newly_added_keys_this_iter = [item[1] for item in vlm_query_candidates if item[1] in pseudo_labels_for_indices and item[1] in unlabeled_indices]
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in newly_added_keys_this_iter]
            
            logger.info(f"VLM added {newly_verified_indices_count} new pseudo-labels this iteration.")
            logger.info(f"Total pseudo-labels: {len(pseudo_labels_for_indices)}. Remaining unlabeled: {len(unlabeled_indices)}")
        else:
            logger.info("VLM did not add any new pseudo-labels this iteration.")
            # Optional: Add convergence criteria (e.g., break if no new samples for X iters)

        # --- 5.E. Logging & Evaluation on Validation Set ---
        if len(processed_val_dataset) > 0 and trainer.eval_dataset:
            logger.info(f"Evaluating model from iteration {iteration + 1} on validation set...")
            eval_metrics = trainer.evaluate(eval_dataset=processed_val_dataset) # trainer.eval_dataset should be processed_val_dataset
            logger.info(f"Validation Metrics Iteration {iteration + 1}: {eval_metrics}")
            if general_config.get('wandb_enabled', False) and 'wandb' in sys.modules:
                import wandb
                wandb_metrics = {f"val_iter_{iteration+1}_{k.replace('eval_', '')}": v for k,v in eval_metrics.items()}
                wandb_metrics[f"iter_{iteration+1}_newly_verified"] = newly_verified_indices_count
                wandb_metrics[f"iter_{iteration+1}_total_pseudo_labels"] = len(pseudo_labels_for_indices)
                wandb_metrics[f"iter_{iteration+1}_unlabeled_pool_size"] = len(unlabeled_indices)
                wandb.log(wandb_metrics, step=iteration + 1)
        else:
            logger.info("Skipping validation set evaluation as it's empty or not provided to trainer.")


    # --- 6. Final Evaluation on Test Set ---
    logger.info("VLM iterations finished. Performing final evaluation on the test set...")
    final_model = trainer.model 
    if len(processed_test_dataset) > 0 and trainer.eval_dataset : # Check if test set can be evaluated
        test_metrics = trainer.evaluate(eval_dataset=processed_test_dataset, metric_key_prefix="final_test")
        logger.info(f"Final Test Metrics: {test_metrics}")
        if general_config.get('wandb_enabled', False) and 'wandb' in sys.modules:
            import wandb
            wandb.log({f"final_test_{k.replace('eval_', '')}": v for k,v in test_metrics.items()})
    else:
        logger.info("Skipping final test set evaluation as it's empty or not provided.")

    logger.info(f"VLM-ITL pipeline finished. Results in {output_dir}")
    if general_config.get('wandb_enabled', False) and 'wandb' in sys.modules:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        #config_file_path = sys.argv[1]
        config_file_path = "configs/vlm_itl_config.yaml" # For testing
        # Basic check if config file exists
        if not os.path.isfile(config_file_path):
            logger.error(f"Configuration file not found: {config_file_path}")
            sys.exit(1)
        run_vlm_itl_pipeline(config_file_path)
    else:
        logger.error("Please provide the path to the configuration file as a command-line argument.")
        logger.info("Example: python scripts/train_vlm_itl.py configs/vlm_itl_config.yaml")
        sys.exit(1)