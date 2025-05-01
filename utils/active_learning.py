import random
import math
import logging
from typing import List, Tuple, Dict, Optional, Callable, Any

import torch
import torch.nn.functional as F
import numpy as np
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances
from datasets import Dataset, concatenate_datasets
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from PIL import Image

from .vlm import VLMHandler # Import VLM handler
from .log_utils import log_active_learning_summary # Import summary logger


logger = logging.getLogger(__name__)

def pad_collate(batch):
    """
    Pads all 'pixel_values' in the batch to the same H×W, and stacks __orig_idx__.
    Assumes each item is a dict with keys:
      - 'pixel_values': Tensor[C, H_i, W_i]
      - '__orig_idx__': int
      - possibly other keys which we ignore here
    """
    # find max H and W
    heights = [item['pixel_values'].shape[1] for item in batch]
    widths  = [item['pixel_values'].shape[2] for item in batch]
    max_h, max_w = max(heights), max(widths)

    pix_vals, orig_idxs = [], []
    for item in batch:
        pv = item['pixel_values']
        c, h, w = pv.shape
        pad_h = max_h - h
        pad_w = max_w - w
        # pad (left, right, top, bottom)
        pv = F.pad(pv, (0, pad_w, 0, pad_h), value=0)
        pix_vals.append(pv)
        orig_idxs.append(item['__orig_idx__'])

    return {
      'pixel_values': torch.stack(pix_vals, dim=0),
      '__orig_idx__': torch.tensor(orig_idxs, dtype=torch.long),
    }


def sample_initial_data(
    full_train_dataset: Dataset,
    initial_percentage: float,
    seed: int
) -> Tuple[Dataset, List[int]]:
    """
    Samples the initial subset of the training data.

    Args:
        full_train_dataset (Dataset): The complete training dataset.
        initial_percentage (float): The fraction (0.0 to 1.0) of data to sample initially.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[Dataset, List[int]]:
            - The initial sampled training dataset.
            - A list of indices used for the initial sample.
    """
    if not 0.0 < initial_percentage <= 1.0:
        raise ValueError("initial_percentage must be between 0 (exclusive) and 1 (inclusive).")

    num_total_samples = len(full_train_dataset)
    num_initial_samples = math.ceil(num_total_samples * initial_percentage)
    num_initial_samples = min(num_initial_samples, num_total_samples) # Ensure not more than available

    logger.info(f"Sampling initial {num_initial_samples} ({initial_percentage:.1%}) data points from {num_total_samples}.")

    # Set seed for shuffling/sampling
    np.random.seed(seed)
    all_indices = list(range(num_total_samples))
    np.random.shuffle(all_indices) # Shuffle indices in place

    initial_indices = all_indices[:num_initial_samples]
    sampled_dataset = full_train_dataset.select(initial_indices)

    logger.info(f"Initial dataset created with {len(sampled_dataset)} samples.")
    return sampled_dataset, initial_indices

def build_pool(dataset, indices, preprocess_fn, batch_size):
    """
    Constructs a DataLoader over the given indices after preprocessing.
    """
    pool = dataset.select(indices)
    pool = pool.add_column("__orig_idx__", indices)
    pool_proc = pool.map(preprocess_fn, batched=True)
    pool_proc.set_format("torch")
    loader = DataLoader(
        pool_proc, 
        batch_size=batch_size,
        collate_fn=pad_collate,
    )
    return pool, loader


def compute_lc_uncertainty(model, loader, pool, device):
    """
    Computes pixel-averaged least-confidence scores.
    """
    model.to(device).eval()
    max_probs = []
    orig_idxs = []

    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k not in pool.column_names + ["__orig_idx__"]}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            if probs.ndim == 4:  # segmentation
                m, _ = probs.max(dim=1)          # [B,H,W]
                m = m.view(m.size(0), -1).mean(dim=1)
            else:
                m, _ = probs.max(dim=1)
            max_probs.append(m.cpu())
            orig_idxs.extend(batch["__orig_idx__"])

    maxp = torch.cat(max_probs).numpy()
    uncertainties = 1.0 - maxp
    return np.array(orig_idxs), uncertainties


def compute_bald_uncertainty(model, loader, pool, device, mc_iterations):
    """
    Computes BALD scores via MC-dropout.
    """
    model.to(device).train()  # enable dropout
    all_probs = []
    for _ in range(mc_iterations):
        iter_probs = []
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items()
                      if k not in pool.column_names + ["__orig_idx__"]}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            if probs.ndim == 4:
                probs = probs.permute(0,2,3,1).reshape(-1, probs.size(1))
            iter_probs.append(probs.cpu())
        all_probs.append(torch.cat(iter_probs, dim=0))

    all_probs = torch.stack(all_probs, dim=0)  # [T, M, C]
    p_bar = all_probs.mean(dim=0)
    H_bar = -(p_bar * torch.log(p_bar + 1e-12)).sum(dim=1)
    H_each = -(all_probs * torch.log(all_probs + 1e-12)).sum(dim=2)
    H_exp = H_each.mean(dim=0)
    mi = (H_bar - H_exp).numpy()

    N = len(pool)
    P = mi.shape[0] // N
    uncertainties = mi.reshape(N, P).mean(axis=1)
    return np.array(pool["__orig_idx__"]), uncertainties


def extract_features(model, dataset, indices, preprocess_fn, feature_extractor_fn, batch_size, device):
    """
    Extracts embedding vectors for given indices.
    """
    subpool = dataset.select(indices)
    subpool_proc = subpool.map(preprocess_fn, batched=True)
    subpool_proc.set_format("torch")
    loader = DataLoader(
        subpool_proc, 
        batch_size=batch_size,
        collate_fn=pad_collate,
    )
    feats = []
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch.items()
                  if k not in subpool.column_names}
        out = feature_extractor_fn(model, inputs)
        if out.ndim == 4:
            out = out.mean(dim=[2,3])
        feats.append(out.cpu().numpy())
    return np.vstack(feats)


def k_center_greedy(feats, select_count):
    """
    Core-set k-Center Greedy selection from feature matrix.
    """
    N = feats.shape[0]
    centers = [0]
    if select_count > 1:
        dists = pairwise_distances(feats, feats)
        for _ in range(1, select_count):
            dist_to_centers = dists[:, centers].min(axis=1)
            centers.append(int(np.argmax(dist_to_centers)))
    return centers

def feature_extractor_fn(model, inputs):
    """
    Given a Segformer model and a batch of inputs, returns encoder feature maps
    for k-Center Greedy.
    """
    # inputs must contain 'pixel_values' tensor
    outputs = model.segformer.encoder(
        pixel_values=inputs['pixel_values'],
        return_dict=True
    )
    # last_hidden_state: [B, D, H, W]
    return outputs.last_hidden_state

def select_next_batch_indices(
    available_indices,
    num_to_select,
    strategy="random",
    model=None,
    dataset=None,
    preprocess_fn=None,
    feature_extractor_fn=None,
    batch_size=4,
    device="cuda",
    mc_iterations=5,
    diversify_pool_factor=10,
):
    """
    Chooses next batch using 'random', 'lc', or 'bald_diversity'.
    """
    logger.info(f"Selecting {num_to_select} indices using strategy: {strategy}")
    num_available = len(available_indices)
    num_to_select = min(num_to_select, num_available)
    if num_to_select <= 0:
        return []

    if strategy == "random":
        return available_indices[:num_to_select]

    pool, loader = build_pool(dataset, available_indices, preprocess_fn, batch_size)

    if strategy == "lc":
        orig_idxs, uncs = compute_lc_uncertainty(model, loader, pool, device)
        order = np.argsort(-uncs)
        return orig_idxs[order][:num_to_select].tolist()

    if strategy == "bald_diversity":
        orig_idxs, uncs = compute_bald_uncertainty(model, loader, pool, device, mc_iterations)
        order = np.argsort(-uncs)
        ranked = orig_idxs[order]
        K = min(diversify_pool_factor * num_to_select, num_available)
        topK = ranked[:K]
        feats = extract_features(model, dataset, topK, preprocess_fn,
                                 feature_extractor_fn, batch_size, device)
        centers = k_center_greedy(feats, num_to_select)
        return np.array(topK)[centers].tolist()

    raise ValueError(f"Unknown strategy: {strategy}")

class ActiveLearningProgressCallback(TrainerCallback):
    """A custom Trainer callback to log progress within an active learning loop."""
    def __init__(self, total_al_steps: int, current_al_step: int, current_data_percentage: float):
        self.total_al_steps = total_al_steps
        self.current_al_step = current_al_step
        self.current_data_percentage = current_data_percentage

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Adds active learning context to logs."""
        if state.is_world_process_zero and logs is not None: # Log only on main process
             logs["al_step"] = f"{self.current_al_step}/{self.total_al_steps}"
             logs["al_data_percentage"] = self.current_data_percentage
             # Note: wandb integration in Trainer automatically picks up these logs

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Optionally log AL progress at epoch end too."""
        if state.is_world_process_zero:
            epoch_log = {
                "al_step": f"{self.current_al_step}/{self.total_al_steps}",
                "al_data_percentage": self.current_data_percentage,
                "epoch": state.epoch # Log current epoch within AL step
            }
            # Use wandb.log if available and configured
            if args.report_to and "wandb" in args.report_to and wandb.run:
                 wandb.log(epoch_log, step=state.global_step)

class SegmentationImageLoggerCallback(TrainerCallback):
    """
    A Hugging Face Trainer callback that logs a few input images,
    ground-truth masks, and model predictions to Weights & Biases
    during evaluation (and optionally during training).
    """
    def __init__(
        self,
        processor,
        id2label: Dict[int, str],
        num_samples: int = 4,
        log_train: bool = False
    ):
        """
        Args:
            processor: the SegformerImageProcessor used to prepare pixel_values.
            id2label: mapping class ID → class name.
            num_samples: how many examples to log each time.
            log_train: if True, also log during training.
        """
        self.processor = processor
        self.id2label = id2label
        self.num_samples = num_samples
        self.log_train = log_train

        # will be set by user after Trainer instantiation:
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None

    def _log_batch(self, images, gt_masks, step, tag):
        model = self.trainer.model
        device = next(model.parameters()).device

        inputs = self.processor(images, return_tensors="pt").to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits  # (B, C, H', W')

        # upsample
        target_h, target_w = images[0].size[1], images[0].size[0]
        up = torch.nn.functional.interpolate(
            logits,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False
        )
        preds = up.argmax(dim=1).cpu().numpy()

        wb_images = []
        for img, gt, pred in zip(images, gt_masks, preds):
            if gt.mode != "P":
                gt = gt.convert("P")
            gt_np = np.array(gt, dtype=np.uint8)
            wb_images.append(
                wandb.Image(
                    img,
                    masks={
                        "ground_truth": {
                            "mask_data": gt_np,
                            "class_labels": self.id2label
                        },
                        "prediction": {
                            "mask_data": pred.astype(np.uint8),
                            "class_labels": self.id2label
                        }
                    }
                )
            )

        wandb.log({ tag: wb_images }, step=step+1)  # step+1 to avoid logging at step 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.trainer is None:
            return
        ds = self.eval_dataset
        n = min(self.num_samples, len(ds))
        samples = ds.select(range(n))
        pixel_values = [s["pixel_values"] for s in samples]
        images = [Image.fromarray(pv.numpy().astype(np.uint8).transpose(1, 2, 0)) for pv in pixel_values]
        labels = [s["labels"] for s in samples]
        masks = [Image.fromarray(l.numpy().astype(np.uint8)) for l in labels]
        self._log_batch(images, masks, state.global_step, "eval_samples")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not self.log_train or self.trainer is None or logs is None or "loss" not in logs:
            return
        ds = self.train_dataset
        n = min(self.num_samples, len(ds))
        samples = ds.select(range(n))
        pixel_values = [s["pixel_values"] for s in samples]
        images = [Image.fromarray(pv.numpy().astype(np.uint8).transpose(1, 2, 0)) for pv in pixel_values]
        labels = [s["labels"] for s in samples]
        masks = [Image.fromarray(l.numpy().astype(np.uint8)) for l in labels]
        self._log_batch(images, masks, state.global_step, "train_samples")


# Placeholder for VLM Feedback Logging within AL loop (if needed beyond final summary)
def log_vlm_iteration_summary(vlm_feedback_list: List[Dict[str, Any]], current_step: int):
    """Logs summary statistics about VLM feedback for the current AL iteration."""
    if not vlm_feedback_list or wandb.run is None:
        return

    num_samples = len(vlm_feedback_list)
    vlm_agreements = sum(f['vlm_agrees_with_gt'] for f in vlm_feedback_list)
    actual_correct = sum(f['is_segmentation_correct'] for f in vlm_feedback_list)

    agreement_rate = vlm_agreements / num_samples if num_samples > 0 else 0
    actual_accuracy = actual_correct / num_samples if num_samples > 0 else 0

    log_data = {
        f"VLM_Agreement_Rate": agreement_rate,
        f"Actual_Accuracy_Sampled": actual_accuracy,
        f"VLM_Samples": num_samples
    }
    wandb.log(log_data)
    logger.info(f"AL Step {current_step}: VLM Agreement Rate: {agreement_rate:.3f}, Actual Accuracy (Sampled): {actual_accuracy:.3f}")

if __name__ == "__main__":
    # Example Usage
    print("Testing active learning utilities...")
    # Create a dummy dataset
    dummy_data = [{"id": i, "feature": i * 2} for i in range(100)]
    full_dataset = Dataset.from_list(dummy_data)
    print(f"Full dataset size: {len(full_dataset)}")

    # 1. Test initial sampling
    initial_perc = 0.2
    seed = 42
    initial_ds, initial_idx = sample_initial_data(full_dataset, initial_perc, seed)
    print(f"Initial sampled dataset size: {len(initial_ds)}")
    print(f"Initial indices (first 10): {initial_idx[:10]}")
    assert len(initial_ds) == math.ceil(100 * initial_perc)
    assert len(initial_idx) == len(initial_ds)

    # 2. Test next batch selection
    all_indices_shuffled = list(range(100))
    random.seed(seed)
    random.shuffle(all_indices_shuffled) # Simulate initial shuffle
    initial_indices_set = set(initial_idx)
    remaining_indices = [idx for idx in all_indices_shuffled if idx not in initial_indices_set]
    print(f"Remaining available indices: {len(remaining_indices)}")

    num_next = 30
    next_batch_idx = select_next_batch_indices(remaining_indices, num_next, strategy="random")
    print(f"Selected next batch size: {len(next_batch_idx)}")
    print(f"Next batch indices (first 10): {next_batch_idx[:10]}")
    assert len(next_batch_idx) == num_next
    assert all(idx in remaining_indices for idx in next_batch_idx)
    assert len(set(next_batch_idx)) == num_next # Ensure unique indices

    # Combine datasets (simulate adding data)
    next_batch_ds = full_dataset.select(next_batch_idx)
    combined_ds = concatenate_datasets([initial_ds, next_batch_ds])
    print(f"Combined dataset size: {len(combined_ds)}")
    assert len(combined_ds) == len(initial_ds) + len(next_batch_ds)

    print("\nActive learning utility tests passed.")