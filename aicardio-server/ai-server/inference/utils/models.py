import importlib


__all__ = ["load_model_from_module"]


def load_model_from_module(module_name, model_class, device="cpu"):
    r"""Load model from module.
    
    Args:
        module_name (str): Module to load model.
        model_class (str): Model to load from module.
        device (str): Device to load model
    Returns:
        torch.nn.Module: Loaded PyTorch model
    """
    module = importlib.import_module(module_name)
    model = getattr(module, model_class)().to(device)
    return model