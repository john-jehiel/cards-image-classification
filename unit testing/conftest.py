import pytest
import torch
from test_layers import cnn_layers, efficientnet1_layers, efficientnet2_layers

def pytest_addoption(parser):
    parser.addoption(
        "--model", action="store", default="cnn", help="Select model: cnn, efficientnet1, or efficientnet2"
    )

@pytest.fixture
def input_tensor():
    return torch.randn(1, 3, 224, 224)

def pytest_generate_tests(metafunc):
    if "layer" in metafunc.fixturenames and "layer_name" in metafunc.fixturenames:
        model = metafunc.config.getoption("model").lower()
        if model == "cnn":
            layers = cnn_layers
        elif model == "efficientnet1":
            layers = efficientnet1_layers
        elif model == "efficientnet2":
            layers = efficientnet2_layers
        else:
            raise ValueError(f"Unknown model: {model}")
        metafunc.parametrize("layer,layer_name", layers)
