import wandb
import os
from typing import Dict, Any, Optional
import logging
import matplotlib.pyplot as plt
import sys
import torch
import numpy as np

from data.pascal_voc import (PASCAL_VOC_IGNORE_INDEX)
# Ensure matplotlib is in non-interactive mode
plt.switch_backend('Agg')  # Use 'Agg' backend for non-interactive plotting

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_wandb(config: Dict[str, Any], run_name: str, project_name: Optional[str] = None) -> Optional[wandb.sdk.wandb_run.Run]:
    """Initializes Weights & Biases logging if enabled in config.

    Args:
        config (Dict[str, Any]): The experiment configuration dictionary.
                                 Should contain 'log_with', potentially 'project_name'.
        run_name (str): The name for this specific run.
        project_name (Optional[str]): The W&B project name. Overrides config if provided.

    Returns:
        Optional[wandb.sdk.wandb_run.Run]: The wandb run object if initialized, otherwise None.
    """
    log_provider = config.get('log_with', 'none').lower()

    if log_provider == 'wandb':
        try:
            # Ensure WANDB_API_KEY is set in the environment or user is logged in
            if "WANDB_API_KEY" not in os.environ and wandb.login() is False:
                 logger.warning("WANDB_API_KEY not set and wandb login failed. Disabling wandb.")
                 return None

            resolved_project_name = project_name or config.get('project_name', 'default-segmentation-project')
            logger.info(f"Initializing wandb run '{run_name}' in project '{resolved_project_name}'")

            run = wandb.init(
                project=resolved_project_name,
                name=run_name,
                config=config, # Log the entire config
                reinit=True, # Allow re-initialization in loops (like active learning),
                mode=os.getenv("WANDB_MODE", "online") # Use online or offline mode based on environment variable
            )
            logger.info(f"Wandb initialized successfully. Run page: {run.url}")
            return run
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}", exc_info=True)
            return None
    elif log_provider == 'tensorboard':
        logger.info("TensorBoard logging is selected (integration via Trainer).")
        # Trainer handles TensorBoard automatically if installed and specified in TrainingArguments
        return None
    else:
        logger.info("Logging disabled.")
        return None

def log_active_learning_summary(overall_metrics: Dict[int, Dict[str, float]], config: Dict[str, Any]):
    """Logs a summary table and plots for the active learning process to wandb.

    Args:
        overall_metrics (Dict[int, Dict[str, float]]):
            A dictionary where keys are data percentages (e.g., 10, 20)
            and values are dictionaries of metrics for that stage.
            Example: {10: {'eval_loss': 0.5, 'eval_mean_iou': 0.6}, 20: {...}}
        config (Dict[str, Any]): The experiment configuration.
    """
    log_provider = config.get('log_with', 'none').lower()
    if log_provider == 'wandb' and wandb.run is not None:
        logger.info("Logging active learning summary to wandb.")
        try:
            # 1. Log Summary Table
            table_data = []
            metric_keys = []
            if overall_metrics:
                # Get metric keys from the first entry, assuming consistency
                first_metrics = next(iter(overall_metrics.values()))
                metric_keys = list(first_metrics.keys())

            for percentage, metrics in sorted(overall_metrics.items()):
                row = [percentage] + [metrics.get(key, None) for key in metric_keys]
                table_data.append(row)

            columns = ["Data Percentage"] + metric_keys
            summary_table = wandb.Table(data=table_data, columns=columns)
            wandb.log({"Active Learning Summary": summary_table})

            # 2. Log Line Plots for each metric vs. Data Percentage
            percentages = sorted(overall_metrics.keys())
            for key in metric_keys:
                 # Check if metric values are numeric before plotting
                values = [overall_metrics[p].get(key) for p in percentages if isinstance(overall_metrics[p].get(key), (int, float))]
                valid_percentages = [p for p in percentages if isinstance(overall_metrics[p].get(key), (int, float))]
                if values: # Only plot if there are numeric values
                    wandb.log({f"AL/{key}_vs_Data": wandb.plot.line_series(
                        xs=valid_percentages,
                        ys=[values],
                        keys=[key],
                        title=f"{key} vs. Data Percentage",
                        xname="Data Percentage (%)"
                    )})
            logger.info("Active learning summary logged successfully.")
        except Exception as e:
            logger.error(f"Failed to log active learning summary to wandb: {e}", exc_info=True)
    else:
        logger.info("Wandb not enabled or not initialized. Skipping active learning summary logging.")
        
def debug_log_and_plot(images: torch.Tensor,
                       masks:  torch.Tensor,
                       class_names: list[str],
                       out_path:  str = "debug_segmentation.png") -> None:
    """Helper function to log and plot images and masks for debugging.

    Args:
        images (torch.Tensor): _images_ tensor of shape [B, C, H, W].
        masks (torch.Tensor): _masks_ tensor of shape [B, H, W].
        class_names (list[str]): _class_names_ list of class names.
        out_path (str, optional): _out_path_ path to save the debug image. Defaults to "debug_segmentation.png".
    Returns:
        None
    """
    logger.info("Debugging: Logging and plotting images and masks.")
    if images.dim() != 4 or masks.dim() != 3:
        logger.error("Invalid dimensions for images or masks.")
        raise ValueError("Images must be 4D [B, C, H, W] and masks must be 3D [B, H, W].")
    num_images = images.shape[0]
    if num_images > 4:
        logger.warning("More than 4 images provided. Only the first 4 will be plotted.")
        num_images = 4
    fig, axes = plt.subplots(num_images, 3, figsize=(10, num_images * 5))
    for i in range(num_images):
        # Plot image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Image {i + 1}")
        # Write label
        label = masks[i].cpu().numpy()
        label = np.unique(label)
        label = [class_names[l] for l in label if l != PASCAL_VOC_IGNORE_INDEX]
        axes[i, 1].imshow(img)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"Label {i + 1}: {', '.join(label)}")
        # Plot mask
        mask = masks[i].cpu().numpy()
        axes[i, 2].imshow(mask, cmap='jet', alpha=0.5)
        axes[i, 2].axis('off')
        axes[i, 2].set_title(f"Mask {i + 1}")
        # Add color legend
        for j, class_name in enumerate(class_names):
            axes[i, 2].add_patch(plt.Rectangle((0, 0), 1, 1, color=plt.cm.jet(j / len(class_names)), label=class_name))
        axes[i, 2].legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize='small')
    # Save figure
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"\tWrote debug figure to {out_path}\n")
    sys.exit(1)  # Exit after saving the figure
    return None

if __name__ == '__main__':
    # Example Usage
    dummy_cfg_al = {'log_with': 'wandb', 'project_name': 'test_logging'}
    dummy_run_name = 'test_run'

    # Mock wandb if API key isn't set, to allow running the example
    if "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_MODE"] = "offline" # Run offline if no key
        logger.warning("WANDB_API_KEY not found. Running wandb in offline mode for example.")

    run = setup_wandb(dummy_cfg_al, dummy_run_name)

    if run:
        print(f"Wandb run initialized: {run.name}, URL (offline): {run.dir}")
        # Example logging active learning data
        mock_metrics = {
            10: {'eval_loss': 0.8, 'eval_mean_iou': 0.3, 'eval_accuracy': 0.5},
            20: {'eval_loss': 0.6, 'eval_mean_iou': 0.45, 'eval_accuracy': 0.6},
            30: {'eval_loss': 0.5, 'eval_mean_iou': 0.55, 'eval_accuracy': 0.7}
        }
        log_active_learning_summary(mock_metrics, dummy_cfg_al)
        print("Example active learning summary logged to wandb (check offline run directory).")
        wandb.finish()
    else:
        print("Wandb setup skipped or failed.")