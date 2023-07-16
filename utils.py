import model
import torch
from torch import nn


def load_model(model_path: str,
               class_names: list,
               device) -> nn.Module:
    print("loading model...")
    loaded_model_0 = model.ferCNNV1(input_shape=3,
                                          hidden_units=40,
                                          output_shape=8)

    loaded_model_0.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("loaded model")

    loaded_model_0.to(device)
    # print("\n-----------Model Summary----------")
    # print(loaded_model_0)
    return loaded_model_0
