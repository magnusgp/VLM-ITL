import yaml
from typing import Dict, Any
import os

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the configuration file is not valid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    try:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
        return config
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        raise

if __name__ == '__main__':
    # Example usage: Create a dummy config and test loading
    dummy_config = {
        'project_name': 'test_project',
        'model': {'name': 'test_model', 'layers': 3},
        'training': {'lr': 0.001, 'epochs': 10}
    }
    dummy_path = 'dummy_config.yaml'
    with open(dummy_path, 'w') as f:
        yaml.dump(dummy_config, f)

    try:
        loaded_cfg = load_config(dummy_path)
        print("Config loaded successfully:")
        print(loaded_cfg)
        assert loaded_cfg['model']['layers'] == 3
    except Exception as e:
        print(f"Error during example usage: {e}")
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)