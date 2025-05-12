from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from PIL import Image
import random
import logging

logger = logging.getLogger(__name__)

class VLMHandler(ABC):
    """Abstract Base Class for Vision-Language Model Handlers."""

    @abstractmethod
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the VLM handler with optional configuration."""
        self.config = config or {}
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Loads the underlying VLM."""
        pass

    @abstractmethod
    def ask_binary_question(
        self,
        image: Image.Image,
        segmentation_mask: Optional[Image.Image], # Optional mask visualization
        prompt: str
    ) -> bool:
        """
        Asks a binary (yes/no) question to the VLM based on the image and optionally a mask.

        Args:
            image (Image.Image): The input image.
            segmentation_mask (Optional[Image.Image]): A visualization of the segmentation mask
                                                       (can be overlayed or shown separately).
                                                       May not be used by all VLMs.
            prompt (str): The binary question to ask (e.g., "Is the segmented object a car?").

        Returns:
            bool: The VLM's answer (True for yes, False for no).
        """
        pass

    def get_vlm_feedback(
        self,
        image: Image.Image,
        segmentation_mask: Image.Image, # Assuming mask is always provided for feedback
        predicted_label_name: str,
        ground_truth_label_name: str,
        query_template: str = "Is the primary object in this segmented region correctly identified as {label_name}?"
    ) -> Dict[str, Any]:
        """
        Simulates getting feedback from the VLM about segmentation quality.
        Compares VLM answer about the predicted label to the ground truth.

        Args:
            image (Image.Image): The original image.
            segmentation_mask (Image.Image): The predicted segmentation mask for the region of interest.
                                             This might need preprocessing (e.g., cropping, highlighting).
            predicted_label_name (str): The name of the class predicted by the segmentation model.
            ground_truth_label_name (str): The ground truth class name for comparison.
            query_template (str): A template string for the question asked to the VLM.
                                  Must contain '{label_name}'.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'vlm_query': The actual question asked.
                - 'vlm_prediction_correct': VLM's answer (True/False) to the query about the *predicted* label.
                - 'vlm_agrees_with_gt': Boolean indicating if the VLM's assessment matches the ground truth comparison.
                - 'is_segmentation_correct': Boolean indicating if prediction matches ground truth.
        """
        query = query_template.format(label_name=predicted_label_name)
        vlm_answer = self.ask_binary_question(image, segmentation_mask, query)

        is_correct_segmentation = (predicted_label_name == ground_truth_label_name)
        if random.random() < 0.1: # 10% chance to log the query
            logger.info(f"VLM answer: {vlm_answer}. Predicted label: {predicted_label_name}, GT label: {ground_truth_label_name}.")
        # Does VLM agree with GT?
        # If seg is correct (pred==GT), VLM should say True.
        # If seg is incorrect (pred!=GT), VLM should say False.
        vlm_agrees = (vlm_answer == is_correct_segmentation)

        return {
            "vlm_query": query,
            "vlm_prediction_correct": vlm_answer,
            "vlm_agrees_with_gt": vlm_agrees,
            "is_segmentation_correct": is_correct_segmentation,
        }


class MockVLMHandler(VLMHandler):
    """A Mock VLM Handler that returns random or fixed answers for simulation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the mock handler.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary.
                Can contain 'mock_accuracy' (float, 0.0 to 1.0) to simulate
                a VLM that gives the 'correct' answer (matching GT comparison)
                with a certain probability. Defaults to 0.8 (80% accuracy).
                If 'mock_accuracy' is not provided, defaults to random answers (50%).
        """
        super().__init__(config)
        self.mock_accuracy = self.config.get('mock_accuracy', None)
        if self.mock_accuracy is not None:
             logger.info(f"MockVLM initialized with simulated accuracy: {self.mock_accuracy:.2f}")
        else:
             logger.info("MockVLM initialized with random answers (50% accuracy).")


    def _load_model(self):
        """No actual model to load for the mock handler."""
        logger.debug("MockVLMHandler: No model loading required.")
        pass

    def ask_binary_question(
        self,
        image: Image.Image,
        segmentation_mask: Optional[Image.Image],
        prompt: str
    ) -> bool:
        """Returns a simulated binary answer.

        If mock_accuracy is set, it tries to return the 'correct' answer
        (based on a heuristic comparing prompt content to known labels - simplified here)
        with that probability. Otherwise, returns random True/False.

        *Note*: For accurate simulation based on `mock_accuracy`, the `get_vlm_feedback`
        method is more reliable as it directly compares to ground truth. This method
        provides a simpler random/biased response.
        """
        if self.mock_accuracy is not None:
            # Simple heuristic: Assume 'yes' is more likely if the prompt seems plausible
            # This is very basic and doesn't truly reflect VLM reasoning.
            # A better mock would be integrated within get_vlm_feedback.
            # Let's just return based on probability directly for simplicity here.
             correct_answer_prob = self.mock_accuracy
             # We don't know the 'correct' answer here without GT, so flip a biased coin
             return random.random() < correct_answer_prob # Simplified: higher chance of True if accuracy > 0.5

        else:
             # Purely random answer
             return random.choice([True, False])

    def get_vlm_feedback(
        self,
        image: Image.Image,
        segmentation_mask: Image.Image,
        predicted_label_name: str,
        ground_truth_label_name: str,
        query_template: str = "Is the primary object in this segmented region correctly identified as {label_name}?"
    ) -> Dict[str, Any]:
        """Overrides base method to use mock_accuracy correctly."""
        query = query_template.format(label_name=predicted_label_name)
        is_correct_segmentation = (predicted_label_name == ground_truth_label_name)

        if self.mock_accuracy is not None:
            # Simulate the VLM agreeing with the ground truth situation
            # with probability mock_accuracy
            if random.random() < self.mock_accuracy:
                # VLM gives the "correct" assessment
                vlm_answer = is_correct_segmentation
            else:
                # VLM gives the incorrect assessment
                vlm_answer = not is_correct_segmentation
            vlm_agrees = (vlm_answer == is_correct_segmentation) # Should al1ys be True if random < accuracy
        else:
            # Purely random VLM answer
            vlm_answer = random.choice([True, False])
            vlm_agrees = (vlm_answer == is_correct_segmentation) # Random agreement

        return {
            "vlm_query": query,
            "vlm_prediction_correct": vlm_answer, # VLM's answer re prediction
            "vlm_agrees_with_gt": vlm_agrees, # Did VLM assessment match reality?
            "is_segmentation_correct": is_correct_segmentation, # Reality
        }

class HuggingFaceVLMHandler(VLMHandler):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.pipeline = None

    def _load_model(self):
        try:
            from transformers import pipeline # Ensure pipeline is imported here
            model_name = self.config.get("vlm_model_name", "Salesforce/blip-vqa-base") # Example
            task = "visual-question-answering"
            logger.info(f"Loading VLM pipeline: {model_name} for task: {task}")
            # Note: May need specific device placement (e.g., device=0 for GPU)
            self.pipeline = pipeline(task, model=model_name)
            logger.info("VLM pipeline loaded successfully.")
        except ImportError:
            logger.error("transformers library not installed or failed to import pipeline. Install with `pip install transformers`")
            raise
        except Exception as e:
            logger.error(f"Failed to load VLM pipeline model '{model_name}': {e}", exc_info=True)
            raise

    def ask_binary_question(
        self,
        image: Image.Image,
        segmentation_mask: Optional[Image.Image], # VQA pipelines might not use the mask directly
        prompt: str
    ) -> bool:
        if self.pipeline is None:
             logger.error("VLM pipeline not initialized.")
             raise RuntimeError("VLM pipeline not available.")

        try:
            # Ensure prompt clearly asks a yes/no question
            if not prompt.lower().strip().startswith((
                "is ", "are ", "does ", "do ", "can ", "will ", 
                "should ", "has ", "have ", "was ", "were ", "did ")):
                 logger.warning(f"Prompt '{prompt}' may not be a typical binary question. VLM might not give a clear yes/no.")

            # For most VQA pipelines, only the image and question are standard inputs.
            # If the segmentation_mask is meant to be part of the visual input,
            # it would typically need to be overlaid on the image before this call.
            result = self.pipeline(image, question=prompt, top_k=1) # Get the top answer

            # Process the result to get a boolean answer
            if not result or not isinstance(result, list) or not result[0] or 'answer' not in result[0]:
                logger.error(f"VLM returned an unexpected result format for prompt '{prompt}'. Result: {result}")
                return False # Fallback for unexpected format
            
            answer_text = result[0]['answer'].lower().strip()
            logger.debug(f"VLM raw answer for '{prompt}': {answer_text}")

            # Simple yes/no parsing
            if answer_text.startswith("yes"):
                return True
            elif answer_text.startswith("no"):
                return False
            else:
                logger.warning(f"VLM answer '{answer_text}' to prompt '{prompt}' is not a clear yes/no. Defaulting to False.")
                return False

        except Exception as e:
            logger.error(f"Error during VLM inference for prompt '{prompt}': {e}", exc_info=True)
            return False

def get_vlm_handler(config: Dict[str, Any]) -> VLMHandler:
    """Factory function to get the appropriate VLM handler based on config."""
    vlm_type = config.get("vlm_itl", {}).get("vlm_handler", "mock").lower()
    vlm_options = config.get("vlm_itl", {}).get("vlm_options", {})

    if vlm_type == "mock":
        logger.info("Creating MockVLMHandler.")
        return MockVLMHandler(vlm_options)
    elif vlm_type == "huggingface_blip": # Example for future extension
        logger.info("Creating HuggingFaceVLMHandler (BLIP).")
        # Ensure necessary options like model name are passed if needed
        vlm_options["vlm_model_name"] = vlm_options.get("vlm_model_name", "Salesforce/blip-vqa-base")
        return HuggingFaceVLMHandler(vlm_options)
    else:
        raise ValueError(f"Unsupported VLM handler type: {vlm_type}")


if __name__ == '__main__':
    # Example Usage
    print("Testing MockVLMHandler...")
    mock_config_random = {'vlm_itl': {'vlm_handler': 'mock'}}
    mock_handler_random = get_vlm_handler(mock_config_random)

    mock_config_biased = {'vlm_itl': {'vlm_handler': 'mock', 'vlm_options': {'mock_accuracy': 0.9}}}
    mock_handler_biased = get_vlm_handler(mock_config_biased)

    # Create dummy image/mask
    dummy_img = Image.new('RGB', (60, 30), color = 'red')
    dummy_mask = Image.new('L', (60, 30), color = 1) # Grayscale mask

    # Test ask_binary_question
    prompt1 = "Is this image red?"
    print(f"Random Mock VLM answer to '{prompt1}': {mock_handler_random.ask_binary_question(dummy_img, None, prompt1)}")
    print(f"Biased Mock VLM answer to '{prompt1}': {mock_handler_biased.ask_binary_question(dummy_img, None, prompt1)}")

    # Test get_vlm_feedback
    print("\nTesting VLM Feedback Simulation (Biased Mock VLM, 90% accuracy):")
    feedback_correct_seg = mock_handler_biased.get_vlm_feedback(
        dummy_img, dummy_mask, "car", "car", "Is the object a {label_name}?"
    )
    print(f"Feedback (Correct Segmentation): {feedback_correct_seg}")
    assert feedback_correct_seg["is_segmentation_correct"] is True

    feedback_incorrect_seg = mock_handler_biased.get_vlm_feedback(
        dummy_img, dummy_mask, "bicycle", "car", "Is the object a {label_name}?"
    )
    print(f"Feedback (Incorrect Segmentation): {feedback_incorrect_seg}")
    assert feedback_incorrect_seg["is_segmentation_correct"] is False

    # Count how often the VLM agrees with reality over N trials
    agree_count = 0
    n_trials = 1000
    print(f"\nSimulating {n_trials} feedback calls to check mock accuracy...")
    for _ in range(n_trials):
        # Randomly choose if the segmentation was correct or not for this trial
        is_correct = random.choice([True, False])
        pred_label = "cat" if is_correct else "dog"
        gt_label = "cat"
        feedback = mock_handler_biased.get_vlm_feedback(dummy_img, dummy_mask, pred_label, gt_label)
        if feedback["vlm_agrees_with_gt"]:
            agree_count += 1

    simulated_accuracy = agree_count / n_trials
    print(f"Simulated VLM agreement accuracy: {simulated_accuracy:.3f} (Expected ~{mock_handler_biased.config.get('mock_accuracy', 0.5):.3f})")
    assert abs(simulated_accuracy - mock_handler_biased.config.get('mock_accuracy', 0.5)) < 0.15 # Allow some statistical noise

    print("\nTesting HuggingFaceVLMHandler...")
    try:
        hf_config = {'vlm_itl': {'vlm_handler': 'huggingface_blip'}}
        hf_handler = get_vlm_handler(hf_config)
        hf_handler._load_model()  # Load the model
        print(f"HF VLM answer to '{prompt1}': {hf_handler.ask_binary_question(dummy_img, None, prompt1)}")
    except Exception as e:
        print(f"Failed to create or use HuggingFaceVLMHandler: {e}")