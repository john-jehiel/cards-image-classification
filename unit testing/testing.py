import pytest
import torch
import torch.nn as nn
from thop import profile
from custom_profile import custom_profile

def test_mac_computation(input_tensor, layer, layer_name):
    if isinstance(layer, nn.Linear):
        input_data = torch.randn(1, layer.in_features)  # (batch_size, input_features)
    elif isinstance(layer, nn.Conv2d):
        input_data = torch.randn(1, layer.in_channels, 224, 224)  # (batch_size, channels, height, width)
    elif isinstance(layer, nn.BatchNorm2d):
        input_data = torch.randn(1, layer.num_features, 224, 224)  # (batch_size, num_features, height, width)
    elif isinstance(layer, nn.BatchNorm1d):
        input_data = torch.randn(1, layer.num_features)  # (batch_size, num_features)
    else:
        input_data = input_tensor

    class TestModel(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            return self.layer(x)

    model = TestModel(layer)

    # Compute MACs using the custom function
    custom_macs, _, _ = custom_profile(model, inputs=(input_data,))
    
    # Compute MACs using the built-in THOP function
    thop_macs, _ = profile(model, inputs=(input_data,))

    assert custom_macs == thop_macs, \
        f"MAC mismatch for {layer_name}: Custom={custom_macs}, THOP={thop_macs}"