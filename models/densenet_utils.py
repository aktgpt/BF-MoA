from torch import nn


def get_kwargs(model_name):
    if model_name == "densenet121":
        return {"growth_rate": 32, "block_config": (6, 12, 24, 16), "num_init_features": 64}
    elif model_name == "densenet161":
        return {"growth_rate": 48, "block_config": (6, 12, 36, 24), "num_init_features": 96}
    elif model_name == "densenet169":
        return {"growth_rate": 32, "block_config": (6, 12, 32, 32), "num_init_features": 64}
    elif model_name == "densenet201":
        return {"growth_rate": 32, "block_config": (6, 12, 48, 32), "num_init_features": 64}
