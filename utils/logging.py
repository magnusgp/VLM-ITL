import wandb
import os
from typing import Dict, Any, Optional
import logging

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
                reinit=True # Allow re-initialization in loops (like active learning)
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