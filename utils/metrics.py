import evaluate
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Global metric object (lazy loaded)
_metric = None

def _load_metric():
    """Loads the Mean IoU metric from the evaluate library."""
    global _metric
    if _metric is None:
        try:
            _metric = evaluate.load("mean_iou")
            logger.info("Mean IoU metric loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load 'mean_iou' metric from evaluate library: {e}", exc_info=True)
            raise RuntimeError("Could not load mean_iou metric.") from e
    return _metric

def compute_metrics_segmentation(eval_pred, num_labels: int, ignore_index: int = 255) -> Optional[Dict[str, float]]:
    """
    Computes segmentation metrics, primarily Mean IoU, for the Huggingface Trainer.

    Args:
        eval_pred (EvalPrediction): A tuple containing predictions and labels.
            - predictions: Logits tensor of shape (batch_size, num_labels, height, width).
            - label_ids: Ground truth labels tensor of shape (batch_size, height, width).
        num_labels (int): The total number of segmentation classes.
        ignore_index (int): The label index to ignore during metric calculation (often padding or background).
                            PASCAL VOC typically doesn't use 255 for ignore, background is 0.
                            Check dataset specifics. For ADE20k based models like Segformer, 255 is common.
                            Let's assume 255 might be used by padding during batching.

    Returns:
        Optional[Dict[str, float]]: A dictionary containing calculated metrics (e.g., mean_iou, mean_accuracy),
                                     or None if metric calculation fails.
    """
    try:
        metric = _load_metric()
        logits, labels = eval_pred
        # Logits are usually (batch, num_labels, H, W)
        # Labels are usually (batch, H, W)

        # Get predictions by taking argmax along the class dimension
        predictions = np.argmax(logits, axis=1)

        # Ensure predictions and labels are numpy arrays
        predictions = np.array(predictions, dtype=np.uint8)
        labels = np.array(labels, dtype=np.uint8)

        # Flatten preds and labels for the metric calculation
        pred_flat = predictions.flatten()
        label_flat = labels.flatten()

        # Compute metrics using the evaluate library
        metrics = metric.compute(
            predictions=pred_flat,
            references=label_flat,
            num_labels=num_labels,
            ignore_index=ignore_index,
            reduce_labels=False, # Set to True if labels are not contiguous from 0 to num_labels-1
        )

        # The metric returns dict like:
        # {'mean_iou': 0.xx, 'mean_accuracy': 0.xx, 'overall_accuracy': 0.xx,
        #  'per_category_iou': array([...]), 'per_category_accuracy': array([...])}

        # We mainly care about mean IoU and mean accuracy for reporting
        # Round metrics for cleaner logging
        metrics_log = {
            "mean_iou": round(metrics["mean_iou"], 4),
            "mean_accuracy": round(metrics["mean_accuracy"], 4),
            "overall_accuracy": round(metrics["overall_accuracy"], 4),
        }
        return metrics_log

    except Exception as e:
        logger.error(f"Error during metric computation: {e}", exc_info=True)
        # Return default/empty values or None to avoid crashing Trainer
        return None # Or return {"mean_iou": 0.0, "mean_accuracy": 0.0, "overall_accuracy": 0.0}


if __name__ == '__main__':
    # Example Usage
    print("Testing metric computation...")
    # Mock data (Batch=2, Classes=3, H=4, W=4)
    mock_logits = np.random.rand(2, 3, 4, 4).astype(np.float32)
    mock_labels = np.random.randint(0, 3, size=(2, 4, 4), dtype=np.uint8)
    # Introduce an ignored label index example
    mock_labels[0, 0, 0] = 255

    num_classes = 3
    ignore_idx = 255

    mock_eval_pred = (mock_logits, mock_labels)

    computed_metrics = compute_metrics_segmentation(mock_eval_pred, num_labels=num_classes, ignore_index=ignore_idx)

    if computed_metrics:
        print("Computed Metrics:")
        print(computed_metrics)
        assert "mean_iou" in computed_metrics
        assert "mean_accuracy" in computed_metrics
        assert "overall_accuracy" in computed_metrics
        print("Metric computation test successful.")
    else:
        print("Metric computation test failed.")

    # Test failure case (e.g., metric loading fails)
    # Temporarily sabotage the metric loading
    _original_metric = _metric
    _metric = "Force Error"
    try:
        print("\nTesting metric computation failure...")
        fail_metrics = compute_metrics_segmentation(mock_eval_pred, num_labels=num_classes, ignore_index=ignore_idx)
        assert fail_metrics is None
        print("Failure test successful (returned None as expected).")
    except Exception as e:
         print(f"Caught unexpected exception during failure test: {e}")
    finally:
        _metric = _original_metric # Restore metric