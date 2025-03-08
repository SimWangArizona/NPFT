import copy

import torch
from peft import LoraConfig, get_peft_model, PeftModel
# from torchgen.api.python import argument_type_str
from transformers import AutoModelForCausalLM,LlamaForCausalLM, Trainer,TrainingArguments

from tqdm import tqdm
from datasets import Dataset
# from torch import distributed as dist
from transformers import TrainerCallback, TrainerControl, TrainerState

# from quantization.chunk_models import module_names
# from squeezellm import *
from src.utils import (
    get_layers,
    get_module_names,
    get_modules,
    load_model,
    parse_model,
    get_combined_train_llama,
    
)
from src.outliers import (
    calculate_gradients,
    calculate_outliers
)
from src.noise import(
    add_noise_to_llama,
)

import tracemalloc
import time


def get_c4_train(nsamples=50, seed=0, seqlen=512, model_path = ""):
    train_loader, val_loader = get_c4(nsamples=128, seed=0, seqlen=seqlen, model=model_path)

    return train_loader


# def add_noise_to_tensor(weight,outlier_positions):
#     weight = weight.cuda()
#     outlier_positions = outlier_positions.cuda()
#     unique_rows = torch.unique(outlier_positions[:, 0])
#     for row_idx in unique_rows:
#         # 获取该行的所有列索引
#         cols = outlier_positions[outlier_positions[:, 0] == row_idx][:, 1]

#         # 提取该行中指定列位置的值
#         row_values = weight[row_idx, cols]
#         sum_before = row_values.sum()
#         # print("before adding noise sum is:",sum_before)
#         # 计算最大值和最小值
#         max_val, min_val = row_values.max(), row_values.min()

#         # 生成均匀分布的随机噪声，范围是 [min_val, max_val] and ensure noise are non-differentiable
#         noise = torch.rand_like(row_values,requires_grad=True) * (max_val - min_val) + min_val
#         noise = noise.cuda()
#         # 调整噪声使其均值为 0
#         noise -= noise.mean()

#         # 将噪声添加到指定列的位置
#         weight[row_idx, cols] += noise
#         row_values = weight[row_idx, cols]
#         sum_after = row_values.sum()
#         # print("after adding noise sum is:",sum_after)

#     return weight.cpu()
# def add_noise_to_model(model,outlier_idx):
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
# Manually train
def train_model(initial_model_path,data_loader,lora_config, epochs=3,output_dir = ''):

    lr_scheduler = [5e-5,5e-5,5e-5,5e-6,5e-6,5e-6,5e-7,5e-7,5e-7,5e-7]
    for epoch in range(1,epochs+1):
        tracemalloc.start()
        import os
        # We should use random initialize lora weights for epoch 1
        if epoch == 1:
            print(f"Training epoch {epoch}")
            import pickle

            original_model = load_model(initial_model_path, model_type="llama")
            gradient_model = calculate_gradients(model_to_cal=original_model, tokenizer_path=initial_model_path,
                                                 gradient_save_path=os.path.join(output_dir, f"gradient_epoch{epoch}"),seqlen=args.gradient_seqlen)

            model_outliers_dict = calculate_outliers(model=original_model,
                                                     gradient_model=gradient_model,not_using_threshold=args.not_using_threshold,range=args.range,
                                                     sensitivity=args.sensitivity,model_type=args.model_type)

            with open(os.path.join(output_dir, f"epoch{epoch}outlier_dict_{args.sensitivity}"), 'wb') as f:
                pickle.dump(model_outliers_dict, f)
            torch.cuda.empty_cache()
            original_model = load_model(initial_model_path, model_type="llama")

            # The noise are non derivative and note that the original model will be changed
            noise_model = add_noise_to_llama(model = original_model,outlier_idx=model_outliers_dict)
            torch.cuda.empty_cache()
    
            print(f"Complete adding noise!")

            # build lora from scratch
            model = get_peft_model(model=noise_model,peft_config=lora_config)
            # model = PeftModel.from_pretrained(original_model,os.path.join(output_dir,f"epoch{epoch}lora_weights"),is_trainable=True)
            model.bfloat16()
            training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, f"epoch{epoch}"),
                evaluation_strategy="no",
                learning_rate=lr_scheduler[epoch-1],
                per_device_train_batch_size=args.batch_size,
                num_train_epochs=1,
                weight_decay=0.01,
                deepspeed=None,
                local_rank=-1,
                save_strategy="epoch")

            trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data_loader,
            )

            trainer.train()
            model.save_pretrained(os.path.join(output_dir,f"epoch{epoch}lora_weights_only_sensitivity{args.sensitivity}"))
            
            print(f"Complete training!")

        else:
            if not os.path.exists(os.path.join(output_dir,f"epoch{epoch}outlier_dict_{args.sensitivity}")):
                torch.cuda.empty_cache()
                print(f"Training epoch {epoch}")
                # Use last epoch's lora weights for epoch>1
                print(f"Epoch {epoch}, loading lora weights from last epoch.")
                original_model = load_model(initial_model_path, model_type="llama")
                torch.cuda.empty_cache()
                lora_model = PeftModel.from_pretrained(original_model, os.path.join(output_dir,f"epoch{epoch-1}lora_weights_only_sensitivity{args.sensitivity}"))
                #
                merged_model = lora_model.merge_and_unload()
                del lora_model
                print(f"Epoch {epoch},calculate sensitivity of the merged model.")
                gradient_model= calculate_gradients(model_to_cal=merged_model,tokenizer_path=initial_model_path,gradient_save_path=os.path.join(output_dir,f"gradient_epoch{epoch}"),seqlen=args.gradient_seqlen)
                # gradient_model = load_model("/groups/huanruiyang/dongweiw/controllableQ/LORA_FT/finetuned/results/gradient_epoch2",model_type = "llama")
                del merged_model
                torch.cuda.empty_cache()
                original_model = load_model(initial_model_path, model_type="llama")
                last_epoch_model = PeftModel.from_pretrained(original_model, os.path.join(output_dir,f"epoch{epoch-1}lora_weights_only_sensitivity{args.sensitivity}"))
                model_outliers_dict = calculate_outliers(model=last_epoch_model.merge_and_unload(),
                                                     gradient_model=gradient_model,not_using_threshold=args.not_using_threshold,range=args.range,
                                                     sensitivity=args.sensitivity)
                del gradient_model
                del last_epoch_model
                #
                import pickle
                with open(os.path.join(output_dir,f"epoch{epoch}outlier_dict_{args.sensitivity}"), 'wb') as f:
                    pickle.dump(model_outliers_dict, f)
            else:
                import pickle
                with open(os.path.join(output_dir,f"epoch{epoch}outlier_dict_{args.sensitivity}"), 'rb') as f:
                    model_outliers_dict = pickle.load(f)
            print(f"Adding noise for epoch {epoch}.")
            torch.cuda.empty_cache()
            original_model = load_model(initial_model_path, model_type="llama")
            noise_model = add_noise_to_llama(model=original_model, outlier_idx=model_outliers_dict)
            torch.cuda.empty_cache()
            model = PeftModel.from_pretrained(noise_model, os.path.join(output_dir,f"epoch{epoch-1}lora_weights_only_sensitivity{args.sensitivity}"),is_trainable=True)

            model.bfloat16()
            training_args = TrainingArguments(
                output_dir=os.path.join(output_dir, f"epoch{epoch}"),
                evaluation_strategy="no",
                learning_rate=lr_scheduler[epoch-1],
                per_device_train_batch_size=2,
                num_train_epochs=1,
                weight_decay=0.01,
                deepspeed=None,
                local_rank=-1,
                save_strategy="epoch")

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=data_loader,
                )

            trainer.train()
            model.save_pretrained(os.path.join(output_dir, f"epoch{epoch}lora_weights_only_sensitivity{args.sensitivity}"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="llama model to load")
    parser.add_argument("--model_type", type=str, default="llama", help="model type")

    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    
    parser.add_argument("--output_dir", type=str, default="", help="Trainer output dir")

    
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs.",
    )

    parser.add_argument(
        "--dataset",
        default = 'c4',
        choices=["c4", "wiki"],
        type = str,
        help="load which dataset",
    )
    parser.add_argument(
        "--gradient_seqlen", type=int, default=2048, help="Seqlen of callibration data when calculating gradients."
    )
    parser.add_argument(
        "--finetuning_seqlen", type=int, default=512, help="Seqlen of callibration data when finetuning."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of callibration data when finetuning."
    )
   
    parser.add_argument("--range", type=float,
                        default=0,
                        # required=True,
                        help="threshold for outlier range, e.g. 1.8",
                        )
    parser.add_argument("--not_using_threshold",
                        action="store_true",
                        help="Whether waive threshold outliers", )
    parser.add_argument(
        "--sensitivity", type=float, default=0, help="The fraction of sensitive weights as outliers"
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="rank of lora layers"
    )
    parser.add_argument('--layers',
        default = 'all',
        choices=["all", "self_attn", "mlp"],
        type = str,
        help="Add perturbatio to specific layers")
    args = parser.parse_args()

    DEV = torch.device("cuda:0")

    # 定义 LoRA 配置
    if args.layers == 'all':
        target_modules = ["q_proj","k_proj","v_proj", "o_proj","up_proj","down_proj"]
    elif args.layers == 'self_attn':
        target_modules = ["q_proj","k_proj","v_proj", "o_proj"]
    else:
        target_modules = ["up_proj","down_proj"]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print(f"load {args.dataset} dataset")

    if args.dataset == "c4":
        dataset = get_c4_train(model_path=args.model)

    print("Training!")


    train_model(data_loader=dataset,initial_model_path=args.model,lora_config=lora_config,output_dir=args.output_dir,epochs=args.num_epochs)





