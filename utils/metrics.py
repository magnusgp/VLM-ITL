import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Lazy‑loaded global metric handle
_metric = None


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def _load_metric():
    """
    Lazily load the `mean_iou` metric from the Hugging Face `evaluate` hub.
    """
    global _metric
    if _metric is None:
        try:
            _metric = evaluate.load("mean_iou")
            logger.info("Mean IoU metric loaded successfully.")
        except Exception as e:
            logger.error("Failed to load 'mean_iou' metric: %s", e, exc_info=True)
            raise RuntimeError("Could not load mean_iou metric.") from e
    return _metric


# ────────────────────────────────────────────────────────────────────────────────
# Main entry point for the Trainer
# ────────────────────────────────────────────────────────────────────────────────
def compute_metrics_segmentation(
    eval_pred,
    num_labels: int,
    ignore_index: int = 255,
) -> Optional[Dict[str, float]]:
    """
    Compute Mean IoU / accuracies for semantic‑segmentation predictions.

    HF Trainer passes a tuple (logits, label_ids):
      • logits:  (B, C, h, w) — *may* be lower‑res than labels
      • labels:  (B, H, W)

    We upsample logits to (B, C, H, W) so prediction / reference shapes match
    before feeding them to the `mean_iou` metric.

    Returns a dict with keys: mean_iou, mean_accuracy, overall_accuracy.
    """
    try:
        metric = _load_metric()

        # ── 1. Convert to torch tensors on CPU ────────────────────────────────
        logits, labels = eval_pred

        logits_t = (
            torch.from_numpy(logits) if isinstance(logits, np.ndarray)
            else logits.detach().cpu()
        )
        labels_t = (
            torch.from_numpy(labels) if isinstance(labels, np.ndarray)
            else labels.detach().cpu()
        )

        # ── 2. Upsample logits if their spatial size ≠ labels ────────────────
        if logits_t.shape[-2:] != labels_t.shape[-2:]:
            logits_t = F.interpolate(
                logits_t,
                size=labels_t.shape[-2:],        # match (H, W) of ground truth
                mode="bilinear",
                align_corners=False,
            )

        # ── 3. Arg‑max over the class dimension ───────────────────────────────
        predictions = logits_t.argmax(dim=1).numpy().astype(np.int64)
        labels      = labels_t.numpy().astype(np.int64)

        # ── 4. Build per‑image lists expected by `mean_iou` ───────────────────
        pred_list  = [p for p in predictions]
        label_list = [l for l in labels]

        metrics = metric.compute(
            predictions=pred_list,
            references=label_list,
            num_labels=num_labels,
            ignore_index=ignore_index,
            reduce_labels=False,   # PASCAL VOC labels are already 0‑based
        )

        # ── 5. Round for neat logging ────────────────────────────────────────
        return {
            "mean_iou":        round(metrics["mean_iou"],        4),
            "mean_accuracy":   round(metrics["mean_accuracy"],   4),
            "overall_accuracy": round(metrics["overall_accuracy"], 4),
        }

    except Exception as e:
        logger.error("Error during metric computation: %s", e, exc_info=True)
        # Avoid crashing the Trainer — return zeros so training can continue
        return {
            "mean_iou": 0.0,
            "mean_accuracy": 0.0,
            "overall_accuracy": 0.0,
        }


# ────────────────────────────────────────────────────────────────────────────────
# Quick self‑test (optional)
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running self‑test for compute_metrics_segmentation…")

    # Mock batch: B=2, C=3, h=w=8; labels H=W=16 (lower‑res logits)
    mock_logits = np.random.rand(2, 3, 8, 8).astype(np.float32)
    mock_labels = np.random.randint(0, 3, size=(2, 16, 16), dtype=np.uint8)
    mock_labels[0, 0, 0] = 255  # ignored pixel

    scores = compute_metrics_segmentation(
        (mock_logits, mock_labels),
        num_labels=3,
        ignore_index=255,
    )
    print("Self‑test OK:", scores)
