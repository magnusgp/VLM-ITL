import numpy as np
from datasets import Dataset, concatenate_datasets
from typing import List, Tuple, Dict, Optional, Callable, Any
import random
import math
import logging
import os

from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl
from .vlm import VLMHandler # Import VLM handler
from .log_utils import log_active_learning_summary # Import summary logger
import wandb


logger = logging.getLogger(__name__)

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


def select_next_batch_indices(
    available_indices: List[int],
    num_to_select: int,
    strategy: str = "random",
    # Add more args here for other strategies (e.g., model, predictions)
) -> List[int]:
    """
    Selects the indices for the next batch of data to add to the training set.

    Args:
        available_indices (List[int]): Indices of data points not yet in the training set.
        num_to_select (int): The number of new data points to select.
        strategy (str): The sampling strategy ('random', 'uncertainty', etc.).
                        Currently only 'random' is implemented.

    Returns:
        List[int]: A list of selected indices.
    """
    num_available = len(available_indices)
    num_to_select = min(num_to_select, num_available) # Cannot select more than available

    if num_to_select <= 0:
        return []

    logger.info(f"Selecting next batch of {num_to_select} indices using '{strategy}' strategy from {num_available} available.")

    if strategy == "random":
        # No need to shuffle available_indices again if it was shuffled initially
        # Just take the next slice. If available_indices is not shuffled, shuffle here.
        # Assuming available_indices is the *remaining* part of the initially shuffled list.
        selected_indices = available_indices[:num_to_select]
    # elif strategy == "uncertainty":
    #     # Placeholder for uncertainty sampling
    #     logger.warning("Uncertainty sampling not yet implemented. Using random.")
    #     selected_indices = random.sample(available_indices, num_to_select)
    else:
        raise ValueError(f"Unknown active learning strategy: {strategy}")

    logger.info(f"Selected {len(selected_indices)} indices for the next batch.")
    return selected_indices


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
        f"AL_Step_{current_step}/VLM_Agreement_Rate": agreement_rate,
        f"AL_Step_{current_step}/Actual_Accuracy_Sampled": actual_accuracy,
        f"AL_Step_{current_step}/VLM_Samples": num_samples
    }
    wandb.log(log_data)
    logger.info(f"AL Step {current_step}: VLM Agreement Rate: {agreement_rate:.3f}, Actual Accuracy (Sampled): {actual_accuracy:.3f}")


# Note: The main active learning loop logic will reside in the scripts
# (train_active_learning.py, train_vlm_itl.py) as it orchestrates
# dataset sampling, trainer initialization, training, and evaluation iteratively.
# This file provides the helper functions for sampling.

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