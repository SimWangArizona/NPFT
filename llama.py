import time

import torch
import torch.nn as nn

from squeezellm.modelutils import *
# from squeezellm.quant import *
from squeezellm.model_parse import (
    parse_model,
    get_layers,
    get_embedding,
    get_norm,
)


def get_model(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM,LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained(model)
    model.seqlen = 2048
    return model

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self, bits, perchannel=False, sym=True, mse=False, norm=2.4, grid=100, maxshrink=.8, trits=False):

        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)
        self.scale = torch.zeros_like(self.scale)

    def _quantize(self, x, scale, zero, maxq):
        if maxq < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = self._quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return self._quantize(x, self.scale, self.zero, self.maxq)

        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')
    model = model.to(dev) # ++++++
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)
                if args.not_quantize_outliers:
                    import pickle
                    with open("/groups/huanruiyang/dongweiw/controllableQ/LORA_FT/finetuned/llama-2-7B/epoch1outlier_dict_0.5",
                              'rb') as f:
                        model_sensitivity_dict = pickle.load(f)
                    outlier_dict = model_sensitivity_dict[i]
                    if "q_proj" in name:
                        print(f"Replacing {name}")
                        outlier_indices = outlier_dict["q"]
                    elif "k_proj" in name:
                        print(f"Replacing {name}")
                        outlier_indices = outlier_dict["k"]
                    elif "v_proj" in name:
                        print(f"Replacing {name}")
                        outlier_indices = outlier_dict["v"]
                    elif "o_proj" in name:
                        print(f"Replacing {name}")
                        outlier_indices = outlier_dict["o"]
                    elif "gate_proj" in name:
                        print(f"Replacing {name}")
                        outlier_indices = outlier_dict["gate"]
                    elif "up_proj" in name:
                        print(f"Replacing {name}")
                        outlier_indices = outlier_dict["up"]
                    else:
                        print(f"Replacing {name}")
                        outlier_indices = outlier_dict["down"]
                    for row, col in outlier_indices:
                        subset[name].weight.data[row, col] = W[row, col]
                print(f"quantization error of layer {i} {name}:", torch.norm(W - subset[name].weight.data, p=2))

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache

# def llama_eval(model, testenc, dev):
#     print("Evaluating ...")
#     model_type = parse_model(model)
#     model = model.to(dev)
#     testenc = testenc.input_ids
#     nsamples = testenc.numel() // model.seqlen
#     # print("nsamples************:", nsamples)
#     use_cache = model.config.use_cache
#     model.config.use_cache = False
#     layers = get_layers(model, model_type)
#     model.model.embed_tokens = model.model.embed_tokens.to(dev)
#     # for i in range(len(embeddings)):
#     #     embeddings[i] = embeddings[i].to(dev)
#
#     layers[0] = layers[0].to(dev)
#
#     dtype = next(iter(model.parameters())).dtype
#     inps = torch.zeros(
#         (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
#     )
#     cache = {"i": 0, "attention_mask": None}
#
#     # class Catcher(nn.Module):
#     #     def __init__(self, module):
#     #         super().__init__()
#     #         self.module = module
#     #
#     #     def forward(self, inp, **kwargs):
#     #         inps[cache["i"]] = inp
#     #         cache["i"] += 1
#     #         cache["attention_mask"] = kwargs["attention_mask"]
#     #         if "position_ids" in kwargs:
#     #             cache["position_ids"] = kwargs["position_ids"]
#     #         raise ValueError
#     class Catcher(nn.Module):
#
#         def __init__(self, module):
#             super().__init__()
#             self.module = module
#
#         def forward(self, inp, **kwargs):
#             inps[cache['i']] = inp
#             cache['i'] += 1
#             cache['attention_mask'] = kwargs['attention_mask']
#             cache['position_ids'] = kwargs['position_ids']
#             raise ValueError
#
#     layers[0] = Catcher(layers[0])
#     for i in range(nsamples):
#         # print(model.seqlen)
#         batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
#         try:
#             model(batch)
#         except ValueError:
#             pass
#     layers[0] = layers[0].module
#
#     layers[0] = layers[0].cpu()
#     model.model.embed_tokens = model.model.embed_tokens.cpu()
#
#     # for i in range(len(embeddings)):
#     #     embeddings[i] = embeddings[i].cpu()
#     torch.cuda.empty_cache()
#
#     outs = torch.zeros_like(inps)
#     attention_mask = cache['attention_mask']
#     position_ids = cache['position_ids']
#     # print(position_ids)
#     # print(type(position_ids))
#     for i in range(len(layers)):
#         print("Layer", i)
#         layer = layers[i].to(dev)
#         layer_error = 0
#         if args.nearest:
#             subset = find_layers(layer)
#             for name in subset:
#                 print(f"{name}")
#                 quantizer = Quantizer()
#                 quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
#                 W = subset[name].weight.data
#                 quantizer.find_params(W, weight=True)
#                 subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)
#                 if args.not_quantize_outliers:
#                     import pickle
#                     with open("/groups/huanruiyang/dongweiw/controllableQ/LORA_FT/finetuned/llama-2-7B/epoch1outlier_dict_0.05",
#                               'rb') as f:
#                         model_sensitivity_dict = pickle.load(f)
#                     outlier_dict = model_sensitivity_dict[i]
#                     if "q_proj" in name:
#                         print(f"Replacing {name}")
#                         outlier_indices = outlier_dict["q"]
#                     elif "k_proj" in name:
#                         print(f"Replacing {name}")
#                         outlier_indices = outlier_dict["k"]
#                     elif "v_proj" in name:
#                         print(f"Replacing {name}")
#                         outlier_indices = outlier_dict["v"]
#                     elif "o_proj" in name:
#                         print(f"Replacing {name}")
#                         outlier_indices = outlier_dict["o"]
#                     elif "gate_proj" in name:
#                         print(f"Replacing {name}")
#                         outlier_indices = outlier_dict["gate"]
#                     elif "up_proj" in name:
#                         print(f"Replacing {name}")
#                         outlier_indices = outlier_dict["up"]
#                     else:
#                         print(f"Replacing {name}")
#                         outlier_indices = outlier_dict["down"]
#                     for row, col in outlier_indices:
#                         subset[name].weight.data[row, col] = W[row, col]
#
#                 print(f"quantization error of layer {i} {name}:", torch.norm(W - subset[name].weight.data, p=2))
#                 layer_error += torch.norm(W - subset[name].weight.data, p=2)
#
#         for j in range(nsamples):
#             if model_type == "opt":
#                 outs[j] = layer(
#                     inps[j].unsqueeze(0),
#                     attention_mask=attention_mask,
#                 )[0]
#             else:
#                 assert model_type in ("llama", "mistral")
#                 outs[j] = layer(
#                     inps[j].unsqueeze(0),
#                     attention_mask=attention_mask,
#                     position_ids=position_ids,
#                 )[0]
#         layers[i] = layer.cpu()
#         del layer
#         torch.cuda.empty_cache()
#         inps, outs = outs, inps
#
#     norm = get_norm(model, model_type)
#     if norm is not None:
#         norm = norm.to(dev)
#     model.lm_head = model.lm_head.to(dev)
#
#     testenc = testenc.to(dev)
#     nlls = []
#     for i in range(nsamples):
#         hidden_states = inps[i].unsqueeze(0)
#         if norm is not None:
#             hidden_states = norm(hidden_states)
#         lm_logits = model.lm_head(hidden_states)
#         shift_logits = lm_logits[:, :-1, :].contiguous()
#         shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
#         loss_fct = nn.CrossEntropyLoss()
#         loss = loss_fct(
#             shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
#         )
#         neg_log_likelihood = loss.float() * model.seqlen
#         nlls.append(neg_log_likelihood)
#     ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
#     print(ppl.item())
#
#     model.config.use_cache = use_cache


# loading quantized checkpoint
def load_quant(model, checkpoint, wbits, include_sparse, topX):
    if (
        "xgen" in checkpoint
        or "opt" in checkpoint # checkpoint corresponds to args.load
        or ("vicuna" in checkpoint and "v1.3" in checkpoint)
        or "llama-2" in checkpoint
        or "mistral" in checkpoint
    ):
        # TODO: this is a hacky solution, will be preperly implemented after all the model checkpoints are updated with
        # the new packing scheme that includes the non-linear weights
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model)
        model = AutoModelForCausalLM.from_config(config)
    else:
        from transformers import LlamaForCausalLM
        # load the original model
        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
    model = model.eval()
    layers = find_layers(model)

    state_dict = torch.load(checkpoint)

    # load sparse thresholds from checkpoint
    if include_sparse:
        num_vals = {}
        for k, v in state_dict.items():
            if "sparse_threshold." in k:
                key = k.replace("sparse_threshold.", "")
                num_vals[key] = v
        for k, v in num_vals.items():
            del state_dict["sparse_threshold." + k]
    else:
        num_vals = None

    # replace layers
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]
    make_quant_lut(
        model, layers, wbits, include_sparse=include_sparse, numvals=num_vals, topX=topX
    )
    del layers

    print("Loading model ...")
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.seqlen = 2048
    print("Done.")

    return model


# function for benchmarking runtime
def benchmark(model, input_ids, check=False):
    model_type = parse_model(model)
    layers = get_layers(model, model_type)

    input_ids = input_ids.to(model.gpus[0] if hasattr(model, "gpus") else DEV)
    torch.cuda.synchronize()

    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    for i, layer in enumerate(layers):
        layer.register_forward_hook(clear_past(i))

    print("Benchmarking ...")

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.0

    def sync():
        if hasattr(model, "gpus"):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i : i + 1],
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, : (i + 1)].reshape((1, -1)),
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            max_memory = max(max_memory, torch.cuda.memory_allocated() / 1024 / 1024)
            if check and i != input_ids.numel() - 1:
                tot += loss(
                    out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)
                ).float()
            cache["past"] = list(out.past_key_values)
            del out
        sync()
        import numpy as np

        print("Median:", np.median(times))
        if check:
            print("PPL:", torch.exp(tot / (input_ids.numel() - 1)).item())
            print("max memory(MiB):", max_memory)


if __name__ == "__main__":
    import argparse
    from squeezellm.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="llama model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Which dataset to use for benchmarking.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[3, 4, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--not_quantize_outliers', action='store_true', help='Whether to quantize sensitive outliers.')
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument("--eval", action="store_true", help="evaluate quantized model.")
    parser.add_argument("--load", type=str, default="", help="Load quantized model.")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Number of tokens to use for benchmarking.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Whether to compute perplexity during benchmarking for verification.",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--torch_profile",
        action="store_true",
        help="Use CUDA profiling tool for timing runs.",
    )
    parser.add_argument(
        "--include_sparse",
        action="store_true",
        help="Whether loaded checkpoint has sparse matrix.",
    )
    parser.add_argument(
        "--num_dense_channels",
        type=int,
        default=10,
        help="Number of dense channel used for hybrid kernel.",
    )

    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Whether loaded multilingual dataset.",
    )

    DEV = torch.device("cuda:0")

    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        print(args.model)
        model = load_quant(
            args.model,
            args.load,
            args.wbits,
            args.include_sparse,
            args.num_dense_channels,
        )
    else:
        # print("step in get_model")
        model = get_model(args.model)
        model.eval()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    if args.benchmark:
        model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, : args.benchmark]

            if args.torch_profile:
                from torch.profiler import profile, record_function, ProfilerActivity

                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as p:
                    benchmark(model, input_ids, check=args.check)
                print(
                    p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
                )
            else:
                benchmark(model, input_ids, check=args.check)

    if args.eval:
        datasets = ["c4"]
        for dataset in datasets:
            print("evluating ", dataset)
            if args.multilingual:
                dataloader, testloader = get_mlg_loaders(
                    dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
                )
            else:
                dataloader, testloader = get_loaders(
                    dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
                )
            print("*********seq_len:",model.seqlen)
            llama_eval(model, testloader, DEV)
