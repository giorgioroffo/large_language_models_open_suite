# Import the configuration classes for each model type
import torch

from models.configuration_file import model_configs


def gr_create_model(model_name=None):
    """
    Dynamically creates and configures a model based on the provided settings,
    selecting the correct AutoModel class and its corresponding tokenizer.
    The model is moved to the appropriate device (GPU if available, otherwise CPU).

    Args:
        model_name (str, optional): The name of the model to load. This should be one of the keys specified in the model_configs
        dictionary. If None, the function will default to the first model name found under the given configuration.

    Returns:
        tuple: A tuple containing the loaded model, its corresponding tokenizer, and the config_id. The model is automatically
        moved to the appropriate device (GPU or CPU).

    Raises:
        ValueError: If the model_name is not found within the predefined model_configs dictionary.
    """

    # Identify the configuration for the given model name
    config_id = None
    for key, config_data in model_configs.items():
        if model_name is not None and model_name in config_data["model_names"]:
            config_id = key
            break

    if config_id is None:
        raise ValueError(f"Model name '{model_name}' not found in any configuration.")

    # Retrieve the selected model's configuration
    config_data = model_configs[config_id]

    # Default to the first model name if model_name was not provided
    if model_name is None:
        model_name = config_data["model_names"][0]

    # Extract configuration data
    config = config_data.get("config")
    additional_configs = config_data.get("additional_configs", {})
    model_class = config_data.get("model_class")
    tokenizer_class = config_data.get("tokenizer_class")

    # Load the model with the specified configuration
    model = model_class.from_pretrained(model_name, config=config, **additional_configs)

    # Load the corresponding tokenizer for the model
    tokenizer = tokenizer_class.from_pretrained(model_name)

    # Check for GPU availability and move the model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log device allocation
    print(f"Model '{model_name}' loaded and moved to {device}.")

    return model, tokenizer, config_id
