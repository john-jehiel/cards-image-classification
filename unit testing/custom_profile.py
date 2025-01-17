import torch
import torch.nn as nn
import numpy as np


def calculate_parameters(param_list):
    total_params = 0
    for p in param_list:
        total_params += torch.DoubleTensor([p.nelement()])
    return total_params

def calculate_zero_ops():
    return torch.DoubleTensor([int(0)])

def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int):
    return np.prod(output_size) * (input_size[1] // groups) * np.prod(kernel_size[2:])

def calculate_norm(input_size):
    return torch.DoubleTensor([2 * input_size])

def calculate_prelu(input_size: torch.Tensor):
    return torch.DoubleTensor([int(input_size)])

def calculate_softmax(batch_size, nfeatures):
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return torch.DoubleTensor([int(total_ops)])

def calculate_avgpool(input_size):
    return torch.DoubleTensor([int(input_size)])

def calculate_adaptive_avg(kernel_size, output_size):
    total_div = 1
    kernel_ops = kernel_size + total_div
    return torch.DoubleTensor([int(kernel_ops * output_size)])

def calculate_linear(in_feature, num_elements):
    return torch.DoubleTensor([int(in_feature * num_elements)])


def count_parameters(layer, x, y):
    total_params = 0
    for p in layer.parameters():
        total_params += torch.DoubleTensor([p.numel()])
    layer.total_params[0] = calculate_parameters(layer.parameters())

def zero_ops(layer, x, y):
    layer.total_ops += calculate_zero_ops()

def count_convNd(layer: nn.modules.conv._ConvNd, x, y: torch.Tensor):
    x = x[0]
    layer.total_ops += calculate_conv2d_flops(
        input_size = list(x.shape),
        output_size = list(y.shape),
        kernel_size = list(layer.weight.shape),
        groups = layer.groups
    )

def count_normalization(layer: nn.modules.batchnorm._BatchNorm, x, y):
    x = x[0]
    flops = calculate_norm(x.numel())
    if (getattr(layer, 'affine', False) or getattr(layer, 'elementwise_affine', False)):
        flops *= 2
    layer.total_ops += flops

def count_prelu(layer, x, y):
    x = x[0]
    nelements = x.numel()
    if not layer.training:
        layer.total_ops += calculate_prelu(nelements)

def count_softmax(layer, x, y):
    x = x[0]
    nfeatures = x.size()[layer.dim]
    batch_size = x.numel() // nfeatures
    layer.total_ops += calculate_softmax(batch_size, nfeatures)

def count_avgpool(layer, x, y):
    num_elements = y.numel()
    layer.total_ops += calculate_avgpool(num_elements)

def count_adap_avgpool(layer, x, y):
    kernel = torch.div(
        torch.DoubleTensor([*(x[0].shape[2:])]),
        torch.DoubleTensor([*(y.shape[2:])])
    )
    total_add = torch.prod(kernel)
    num_elements = y.numel()
    layer.total_ops += calculate_adaptive_avg(total_add, num_elements)

def count_linear(layer, x, y):
    total_mul = layer.in_features
    num_elements = y.numel()
    layer.total_ops += calculate_linear(total_mul, num_elements)

register_hooks = {
    nn.ZeroPad2d: zero_ops,
    nn.Conv1d: count_convNd,
    nn.Conv2d: count_convNd,
    nn.Conv3d: count_convNd,
    nn.ConvTranspose1d: count_convNd,
    nn.ConvTranspose2d: count_convNd,
    nn.ConvTranspose3d: count_convNd,
    nn.BatchNorm1d: count_normalization,
    nn.BatchNorm2d: count_normalization,
    nn.BatchNorm3d: count_normalization,
    nn.LayerNorm: count_normalization,
    nn.InstanceNorm1d: count_normalization,
    nn.InstanceNorm2d: count_normalization,
    nn.InstanceNorm3d: count_normalization,
    nn.PReLU: count_prelu,
    nn.Softmax: count_softmax,
    nn.ReLU: zero_ops,
    nn.ReLU6: zero_ops,
    nn.LeakyReLU: zero_ops,
    nn.MaxPool1d: zero_ops,
    nn.MaxPool2d: zero_ops,
    nn.MaxPool3d: zero_ops,
    nn.AdaptiveMaxPool1d: zero_ops,
    nn.AdaptiveMaxPool2d: zero_ops,
    nn.AdaptiveMaxPool3d: zero_ops,
    nn.AvgPool1d: count_avgpool,
    nn.AvgPool2d: count_avgpool,
    nn.AvgPool3d: count_avgpool,
    nn.AdaptiveAvgPool1d: count_adap_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.AdaptiveAvgPool3d: count_adap_avgpool,
    nn.Linear: count_linear,
    nn.Dropout: zero_ops,
    nn.Sequential: zero_ops
}

def custom_profile(model: nn.Module, inputs):
    handler_collection = {}
    types_collection = set()

    def add_hooks(layer: nn.Module):
        layer.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        layer.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))

        layer_type = type(layer)

        fn = None
        if layer_type in register_hooks:
            fn = register_hooks[layer_type]

        if fn is not None:
            handler_collection[layer] = (
                layer.register_forward_hook(fn),
                layer.register_forward_hook(count_parameters),
            )
        types_collection.add(layer_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(*inputs)

    def dfs_count(module: nn.Module):
        total_ops, total_params = module.total_ops.item(), 0
        layer_info_dict = {}
        for layer_name, layer in module.named_children():
            next_dict = {}
            if layer in handler_collection and not isinstance(
                layer, (nn.Sequential, nn.ModuleList)
            ):
                layer_ops, layer_params = layer.total_ops.item(), layer.total_params.item()
            else:
                layer_ops, layer_params, next_dict = dfs_count(layer)
            layer_info_dict[layer_name] = (type(layer).__name__, layer_ops, layer_params, next_dict) # layer_type, mac_count, param_count, next_nested_layer
            total_ops += layer_ops
            total_params += layer_params
        return total_ops, total_params, layer_info_dict

    total_ops, total_params, layer_info_dict = dfs_count(model)

    # reset model to original status
    model.train(prev_training_status)
    for layer, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        layer._buffers.pop("total_ops")
        layer._buffers.pop("total_params")

    return total_ops, total_params, layer_info_dict
