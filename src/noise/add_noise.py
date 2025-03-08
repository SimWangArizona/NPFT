import torch
from utils import *

def add_noise_to_tensor(weight,outlier_positions):
    weight = weight.cuda()
    outlier_positions = outlier_positions.cuda()
    unique_rows = torch.unique(outlier_positions[:, 0])
    for row_idx in unique_rows:
        # 获取该行的所有列索引
        cols = outlier_positions[outlier_positions[:, 0] == row_idx][:, 1]

        # 提取该行中指定列位置的值
        row_values = weight[row_idx, cols]
        sum_before = row_values.sum()
        # print("before adding noise sum is:",sum_before)
        # 计算最大值和最小值
        max_val, min_val = row_values.max(), row_values.min()

        # 生成均匀分布的随机噪声，范围是 [min_val, max_val] and ensure noise are non-differentiable
        noise = torch.rand_like(row_values,requires_grad=True) * (max_val - min_val) + min_val
        noise = noise.cuda()
        # 调整噪声使其均值为 0
        noise -= noise.mean()

        # 将噪声添加到指定列的位置
        weight[row_idx, cols] += noise
        row_values = weight[row_idx, cols]
        sum_after = row_values.sum()
        # print("after adding noise sum is:",sum_after)

    return weight.cpu()
def add_noise_to_llama(model,outlier_idx):
    # Note that the added noise should be Non-differentiable
    tmp_model = model.bfloat16()
    layers = get_layers(model = tmp_model,model_type="llama")

    for layer_idx,(layer,outlier_dict) in enumerate(zip(layers,outlier_idx)):
        print(f"Adding noise to layer {layer_idx}")
        tmp_model.model.layers[layer_idx].self_attn.q_proj.weight.data = add_noise_to_tensor(layers[layer_idx].self_attn.q_proj.weight.data,outlier_dict['q'])
        tmp_model.model.layers[layer_idx].self_attn.k_proj.weight.data = add_noise_to_tensor(layers[layer_idx].self_attn.k_proj.weight.data,outlier_dict['k'])
        tmp_model.model.layers[layer_idx].self_attn.v_proj.weight.data = add_noise_to_tensor(layers[layer_idx].self_attn.v_proj.weight.data,outlier_dict['v'])
        tmp_model.model.layers[layer_idx].self_attn.o_proj.weight.data = add_noise_to_tensor(layers[layer_idx].self_attn.o_proj.weight.data,outlier_dict['o'])
        tmp_model.model.layers[layer_idx].mlp.gate_proj.weight.data = add_noise_to_tensor(layers[layer_idx].mlp.gate_proj.weight.data,outlier_dict['gate'])
        tmp_model.model.layers[layer_idx].mlp.up_proj.weight.data = add_noise_to_tensor(layers[layer_idx].mlp.up_proj.weight.data,outlier_dict['up'])
        tmp_model.model.layers[layer_idx].mlp.down_proj.weight.data = add_noise_to_tensor(layers[layer_idx].mlp.down_proj.weight.data,outlier_dict['down'])

    return tmp_model
def add_noise_to_opt(model,outlier_idx):
    # Note that the added noise should be Non-differentiable
    tmp_model = model.bfloat16()
    layers = get_layers(model = tmp_model,model_type="opt")

    for layer_idx,(layer,outlier_dict) in enumerate(zip(layers,outlier_idx)):
        print(f"Adding noise to layer {layer_idx}")
        tmp_model.model.decoder.layers[layer_idx].self_attn.q_proj.weight.data = add_noise_to_tensor(layers[layer_idx].self_attn.q_proj.weight.data,outlier_dict['q'])
        tmp_model.model.decoder.layers[layer_idx].self_attn.k_proj.weight.data = add_noise_to_tensor(layers[layer_idx].self_attn.k_proj.weight.data,outlier_dict['k'])
        tmp_model.model.decoder.layers[layer_idx].self_attn.v_proj.weight.data = add_noise_to_tensor(layers[layer_idx].self_attn.v_proj.weight.data,outlier_dict['v'])
        tmp_model.model.decoder.layers[layer_idx].self_attn.out_proj.weight.data = add_noise_to_tensor(layers[layer_idx].self_attn.out_proj.weight.data,outlier_dict['o'])
        tmp_model.model.decoder.layers[layer_idx].fc1.weight.data = add_noise_to_tensor(layers[layer_idx].fc1.weight.data,outlier_dict['up'])
        tmp_model.model.decoder.layers[layer_idx].fc2.weight.data = add_noise_to_tensor(layers[layer_idx].fc2.weight.data,outlier_dict['down'])

    return tmp_model