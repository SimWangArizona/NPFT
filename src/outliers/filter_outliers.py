import torch
from utils import *
import tqdm
def generate_config_per_layer(module_names,model_layer,threshold_range):
    total_params = 0
    total_outliers = 0
    json_data = []
    layer_json = {}
    for name in module_names:
        module_weight = model_layer[name]
        # weights_np = module_weight.detach().numpy()
        weights_np = module_weight.detach().to(torch.float32).numpy()
        q1 = np.quantile(weights_np, 0.25)
        q3 = np.quantile(weights_np, 0.75)
        minimum = q1 - threshold_range * (q3 - q1)
        maximum = q3 + threshold_range * (q3 - q1)
        minimum, maximum = -max(abs(minimum), abs(maximum)), max(
            abs(minimum), abs(maximum)
        )
        # print("minimum and maximum", minimum, maximum)
        num_params = weights_np.shape[0] * weights_np.shape[1]
        num_outliers = (weights_np < minimum).sum() + (weights_np > maximum).sum()
        total_params += num_params
        total_outliers += num_outliers

        # print(name, f"% outlier: {num_outliers / num_params * 100:.3f}%")
        layer_json[name] = maximum

    json_data.append(layer_json)

    # Save the json data


    outlier_percentage = total_outliers / total_params * 100
    o = round(outlier_percentage, 2)
    json_data = {
    "outlier_threshold": o,
    "outlier_config": json_data,
    }
    return json_data
def remove_outliers_by_sensitivity(
    model,
    gradients,
    sensitivity,
):
    module_names = list(model.keys())
    outlier_indices_dict = {}  # 用来存储模块名称和outliers索引的字典
    total_outliers = 0
    total_weights = 0

    def _body(gweight, weight):
        num_outliers = int(gweight.numel() * sensitivity / 100)
        thres = gweight.reshape(-1).topk(k=num_outliers).values[-1]

        t = gweight > thres  # 生成表示 outliers 位置的掩码

        return t, t.sum().item(), t.numel()

    for _name in module_names:
        # print("name:",_name)
        weight = model[_name].to(torch.float)
        # print(weight.shape)
        gweight = gradients[_name].to(torch.float)

        outlier_mask, _total_outliers, _total_weights = _body(gweight, weight)
        outlier_indices = outlier_mask.nonzero(as_tuple=False) # 保存 outliers 的索引
        outlier_indices_dict[_name] = outlier_indices
        total_outliers += _total_outliers
        # print("total outliers:",total_outliers)
        total_weights += _total_weights

    print("num of sensitive outliers",total_outliers)
    print("percent of sensitive outlier:", total_outliers / total_weights * 100)
    return outlier_indices_dict

def remove_outliers_by_threshold(
    model,
    outlier_config,
    outlier_indices_dict,  # to avoid memory leak
):
    module_names = list(model.keys())

    total_outliers = 0
    total_weights = 0

    def _body(weight, thres):
        t = torch.logical_or(
            weight >= thres,
            weight <= -thres,
        )
        return t, t.sum().item(), t.numel()

    for i, name in enumerate(module_names):

        thres = outlier_config[name]
        print("thres:",thres)
        weight = model[name].to(torch.float)
        # weight = model[name]

        outlier_mask, _total_outliers, _total_weights = _body(weight, thres)
        new_outlier_indices = outlier_mask.nonzero(as_tuple=False)  # 获取 outliers 的索引
        if name in outlier_indices_dict:
            # 如果该模块名已有 outliers 索引，合并新旧索引
            combined_indices = torch.cat((outlier_indices_dict[name], new_outlier_indices), dim=0)
            # 去重，按行唯一化
            outlier_indices_dict[name] = torch.unique(combined_indices, dim=0)
        else:
            # 如果该模块名还没有任何 outliers 索引，则直接添加新的
            outlier_indices_dict[name] = new_outlier_indices
        total_weights+=_total_weights
        total_outliers+=_total_outliers
    print("total weights:",total_weights)
    print("total outliers:",total_outliers)
    print("percent outlier :", total_outliers / total_weights * 100)
    sum = 0
    for k,v in outlier_indices_dict.items():
        sum+=len(v)
    print("percent outlier after threshold:", sum / total_weights * 100)
    return outlier_indices_dict
def remove_outliers(
    model,
    sensitivity,
    outlier_config,
    gradients=None,
):
    # model and gradients are dictionary of a layer component
    # where the key is the layer name (e.g. q, k, v) and the value is the weight
    assert isinstance(model, dict)
    assert isinstance(gradients, dict) or gradients is None

    assert outlier_config is not None or sensitivity != 0
    if sensitivity != 0:
        assert gradients is not None

    if sensitivity != 0:
        print("removing outliers by sensitivity")
        outlier_indices_dict = remove_outliers_by_sensitivity(
            model=model,
            gradients=gradients,
            sensitivity=sensitivity,
        )
    else:
        outlier_indices_dict = None

    if outlier_config is not None:
        print("removing outliers by threshold")
        outlier_indices_dict = remove_outliers_by_threshold(
            model=model,
            outlier_config=outlier_config,
            outlier_indices_dict=outlier_indices_dict,
        )

    return outlier_indices_dict

def calculate_outliers(model,gradient_model,not_using_threshold = False,range = 0,sensitivity = 0,model_type = 'opt'):
    layers = get_layers(model, model_type = model_type)
    g_layers = get_layers(gradient_model, model_type = model_type)
    module_names = get_module_names(model_type = model_type)
    model_outliers_dict  = []
    for i, (layer, g_layer) in tqdm(enumerate(zip(layers, g_layers))):
        weight = {}
        g_weight = {}

        modules = get_modules(layer, model_type = model_type)
        g_modules = get_modules(g_layer, model_type = model_type)

        for lin, g_lin, name in zip(modules, g_modules, module_names):
            weight[name] = lin.weight.data.cpu()
            g_weight[name] = g_lin.weight.data.cpu()
        if not not_using_threshold:

            config_layer = generate_config_per_layer(module_names=module_names, model_layer=weight,
                                                     threshold_range=range)

            config_layer = config_layer["outlier_config"][0]
            print(config_layer)
        else:
            config_layer = None
        # onlu return the indices of outliers, not removing the outliers
        outlier_indices_dict = remove_outliers(model=weight, sensitivity=sensitivity, outlier_config=config_layer,
                                          gradients=g_weight)
        model_outliers_dict.append(outlier_indices_dict)

    return model_outliers_dict