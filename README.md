# Image Segmentation with Huggingface, Active Learning, and VLM Simulation

This repository provides a modular and extensible framework for image segmentation tasks using PyTorch, the Huggingface `transformers` and `datasets` libraries, and Weights & Biases (wandb) for experiment tracking. It focuses on the PASCAL VOC dataset and implements three experimental pipelines:

1.  **Theoretical Baseline:** Standard supervised training using a full train/validation/test split.
2.  **Practical Baseline (Active Learning Simulation):** Simulates an active learning scenario where the training data is incrementally increased, mimicking a human-in-the-loop labeling process where more data becomes available over time.
3.  **VLM-ITL (Vision-Language Model In-The-Loop Simulation):** Extends the active learning simulation by incorporating feedback from a Vision-Language Model (VLM) to assess segmentation quality at each stage, simulating VLM-assisted quality control or analysis.

## Features

*   **Modular Structure:** Code is organized into data handling, models, utilities, and training scripts.
*   **Huggingface Integration:** Leverages `datasets` for data loading/processing and `Trainer` for streamlined training and evaluation.
*   **Standard Architecture:** Uses SegFormer (`transformers`) for segmentation, easily adaptable to other models.
*   **Experiment Tracking:** Integrates with Weights & Biases (wandb) for logging metrics, configurations, and visualizations.
*   **Type Hinting & Docstrings:** Code includes comprehensive type annotations and docstrings for clarity and maintainability.
*   **Configuration Driven:** Experiments are controlled via YAML configuration files.
*   **Active Learning Simulation:** Implements random sampling for incremental data acquisition.
*   **VLM Simulation:** Includes an abstraction for VLM interaction with a mock implementation and placeholder for real VLMs. Allows simulating VLM feedback on segmentation quality.

## Repository Structure

```
vlm-hitl/
├── configs/                 # Experiment configuration files (YAML)
│   ├── baseline_config.yaml
│   └── active_learning_config.yaml # Used for both AL and VLM-ITL runs
├── data/                    # Data loading and preprocessing
│   ├── __init__.py
│   └── pascal_voc.py        # PASCAL VOC specific loading and helpers
├── models/                  # Model definitions
│   ├── __init__.py
│   └── segformer.py         # SegFormer loading utility
├── scripts/                 # Runnable training scripts
│   ├── train_baseline.py
│   ├── train_active_learning.py
│   └── train_vlm_itl.py
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── active_learning.py   # AL data sampling helpers
│   ├── config.py            # YAML config loading
│   ├── logging.py           # Wandb setup and logging helpers
│   ├── metrics.py           # Segmentation metric calculation
│   └── vlm.py               # VLM handler abstraction and mock implementation
├── .gitignore
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/magnusgp/VLM-ITL
    cd vlm-hitl
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *Note: Ensure you have a compatible PyTorch version installed for your hardware (CPU/GPU). Check [PyTorch Get Started](https://pytorch.org/get-started/locally/) if needed.*

4.  **Set up Weights & Biases:**
    *   Sign up for a free account at [wandb.ai](https://wandb.ai/).
    *   Log in to your account in your terminal:
        ```bash
        wandb login
        ```
    *   Alternatively, set the `WANDB_API_KEY` environment variable.

## Running Experiments

Experiments are run using the scripts in the `scripts/` directory, each requiring a configuration file.

**Configuration Files:**

*   `configs/baseline_config.yaml`: Controls the standard baseline training.
*   `configs/active_learning_config.yaml`: Controls both the Active Learning simulation and the VLM-ITL simulation. Key sections:
    *   `active_learning`: Parameters like initial data percentage, increment, strategy.
    *   `vlm_itl`: Settings specific to the VLM run, including enabling it (`enabled: True`), choosing the handler (`vlm_handler: "mock"`), and handler options.

**Running Commands:**

1.  **Theoretical Baseline:**
    ```bash
    python scripts/train_baseline.py --config configs/baseline_config.yaml
    ```

2.  **Practical Baseline (Active Learning Simulation):**
    *   Ensure `vlm_itl.enabled` is `False` or commented out in `configs/active_learning_config.yaml`.
    ```bash
    python scripts/train_active_learning.py --config configs/active_learning_config.yaml
    ```

3.  **VLM-ITL Simulation:**
    *   Modify `configs/active_learning_config.yaml`:
        *   Set `vlm_itl.enabled: True`.
        *   Configure `vlm_itl.vlm_handler` (e.g., `"mock"`) and `vlm_itl.vlm_options`.
    ```bash
    python scripts/train_vlm_itl.py --config configs/active_learning_config.yaml
    ```

**Expected Output:**

*   Logs will be printed to the console.
*   If `log_with: "wandb"` is set in the config, experiments will be tracked in your Weights & Biases account.
    *   **Baseline:** A single run logging training and evaluation metrics.
    *   **Active Learning / VLM-ITL:** Multiple runs (one per data percentage iteration) will be created, logging metrics for that stage. A final summary run will log overall performance plots (e.g., Mean IoU vs. Data Percentage) and final test set results. For VLM-ITL, additional metrics related to VLM feedback (like agreement rate) will be logged.
*   Model checkpoints, logs, and evaluation results will be saved locally in the directories specified by `output_dir` (baseline) or `output_dir_prefix` (active learning / VLM-ITL) in the configuration files.

## Customization

*   **Dataset:** Modify `data/pascal_voc.py` or add new files in `data/` to support different datasets. Update configuration accordingly.
*   **Model:** Change the model architecture in `models/segformer.py` (or add new model files) and update the configuration (`model.name`). Ensure compatibility with `Trainer`.
*   **Active Learning Strategy:** Implement new strategies in `utils/active_learning.py` (e.g., uncertainty sampling) and add the strategy name to the configuration.
*   **VLM Handler:** Implement new handlers in `utils/vlm.py` inheriting from `VLMHandler` (e.g., integrating with OpenAI API, specific Huggingface models) and update the configuration (`vlm_itl.vlm_handler`).
*   **Hyperparameters:** Adjust parameters in the YAML configuration files.