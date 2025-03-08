import numpy as np
import torch
from sympy.core.random import shuffle


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )
    trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random

    random.seed(0)
    valenc = []
    # print("nsamples:",nsamples)
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )
    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_c4_multilingual(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    # traindata = load_dataset(
    #     "allenai/c4",
    #     "ja",
    #     split="train",
    # )
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "multilingual/c4-is.tfrecord-00000-of-00128.json.gz"},
        split="train",
    )
    # valdata = load_dataset(
    #     "allenai/c4",
    #     "ja",
    #     split="validation",
    # )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "multilingual/c4-is-validation.tfrecord-00000-of-00001.json.gz"},
        split="validation",
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        # print(_)
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            # print(trainenc.input_ids.shape[0])
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        # print("i:",i)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        # print("inp:",inp)
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=""):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if "ptb" in name:
        if "new" in name:
            return get_ptb_new(nsamples, seed, seqlen, model)
        return get_ptb(nsamples, seed, seqlen, model)
    if "c4" in name:
        if "new" in name:
            return get_c4_new(nsamples, seed, seqlen, model)
        else:
            return get_c4(nsamples, seed, seqlen, model)


def get_c4_mixed_ft(nsamples=128, seed=0, seqlen=2048, model=""):
    from datasets import load_dataset

    # 加载英语和冰岛语数据集
    en_traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    is_traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "multilingual/c4-is.tfrecord-00000-of-00128.json.gz"},
        split="train",
    )
    #
    # 验证集可使用相同方式加载
    en_valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    is_valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "multilingual/c4-is-validation.tfrecord-00000-of-00001.json.gz"},
        split="validation",
    )
    #
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )
    #
    import random

    random.seed(seed)
    en_trainloader = []
    #
    # # 合并英语和冰岛语数据集
    # combined_train_data = en_traindata + is_traindata
    #
    # 创建en训练数据
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(en_traindata) - 1)
            trainenc = tokenizer(en_traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        en_trainloader.append((inp, tar))

    random.seed(seed)
    is_trainloader = []
    #
    # # 合并英语和冰岛语数据集
    # combined_train_data = en_traindata + is_traindata
    #
    # 创建is训练数据
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(is_traindata) - 1)
            trainenc = tokenizer(is_traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        is_trainloader.append((inp, tar))

    # # 合并英语和冰岛语验证集
    # combined_val_data = en_valdata + is_valdata
    #
    # random.seed(0)
    # valenc = []
    # for _ in range(256):
    #     while True:
    #         i = random.randint(0, len(combined_val_data) - 1)
    #         tmp = tokenizer(combined_val_data[i]["text"], return_tensors="pt")
    #         if tmp.input_ids.shape[1] >= seqlen:
    #             break
    #     i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     valenc.append(tmp.input_ids[:, i:j])
    # valenc = torch.hstack(valenc)
    #
    # class TokenizerWrapper:
    #     def __init__(self, input_ids):
    #         self.input_ids = input_ids
    #
    # valenc = TokenizerWrapper(valenc)

    return en_trainloader, is_trainloader
            # trainloader, valenc)

def get_c4_mixed_ft_opt(nsamples=128, seed=0, seqlen1=1024,seqlen2 = 512, model=""):
    from datasets import load_dataset

    # 加载英语和冰岛语数据集
    en_traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    is_traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "multilingual/c4-is.tfrecord-00000-of-00128.json.gz"},
        split="train",
    )
    #
    # 验证集可使用相同方式加载
    en_valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    is_valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "multilingual/c4-is-validation.tfrecord-00000-of-00001.json.gz"},
        split="validation",
    )
    #
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=False, trust_remote_code=True
    )
    #
    import random

    random.seed(seed)
    en_trainloader = []
    #
    # # 合并英语和冰岛语数据集
    # combined_train_data = en_traindata + is_traindata
    #
    # 创建en训练数据
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(en_traindata) - 1)
            trainenc = tokenizer(en_traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen1 - 1)
        j = i + seqlen1
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        en_trainloader.append((inp, tar))

    random.seed(seed)
    is_trainloader = []
    #
    # # 合并英语和冰岛语数据集
    # combined_train_data = en_traindata + is_traindata
    #
    # 创建is训练数据
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(is_traindata) - 1)
            trainenc = tokenizer(is_traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen2:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen2 - 1)
        j = i + seqlen2
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        is_trainloader.append((inp, tar))

    # # 合并英语和冰岛语验证集
    # combined_val_data = en_valdata + is_valdata
    #
    # random.seed(0)
    # valenc = []
    # for _ in range(256):
    #     while True:
    #         i = random.randint(0, len(combined_val_data) - 1)
    #         tmp = tokenizer(combined_val_data[i]["text"], return_tensors="pt")
    #         if tmp.input_ids.shape[1] >= seqlen:
    #             break
    #     i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     valenc.append(tmp.input_ids[:, i:j])
    # valenc = torch.hstack(valenc)
    #
    # class TokenizerWrapper:
    #     def __init__(self, input_ids):
    #         self.input_ids = input_ids
    #
    # valenc = TokenizerWrapper(valenc)

    return en_trainloader, is_trainloader

def get_mlg_loaders(name, nsamples=128, seed=0, seqlen=2048, model=""):
    return get_c4_multilingual(nsamples, seed, seqlen, model)

def create_combined_dataset(en_trainloader, is_trainloader):
    input_ids = []
    labels = []
    data_sources = []  # 用于存储数据源信息

    # 处理 en 数据
    for inp, tar in en_trainloader:
        input_ids.append(inp.squeeze().numpy())  # 将张量转换为 numpy 数组
        labels.append(tar.squeeze().numpy())
        data_sources.append("en")  # 添加数据源信息

    # 处理 is 数据
    for inp, tar in is_trainloader:
        input_ids.append(inp.squeeze().numpy())  # 将张量转换为 numpy 数组
        labels.append(tar.squeeze().numpy())
        data_sources.append("is")  # 添加数据源信息

    # 创建 Hugging Face 的 Dataset 对象
    from datasets import Dataset
    return Dataset.from_dict({
        "input_ids": input_ids,
        "labels": labels,
        "data_sources": data_sources
    })
def get_combined_train_llama(model_path,seqlen = 512):
    en_tl ,is_tl= get_c4_mixed_ft(model = model_path,seqlen=seqlen)
    combined_dataset = create_combined_dataset(en_tl, is_tl)
    return combined_dataset
def get_combined_train_opt(model_path):
    en_tl ,is_tl= get_c4_mixed_ft_opt(model = model_path,seqlen1=1024,seqlen2 = 512)
    combined_dataset = create_combined_dataset(en_tl, is_tl)
    return combined_dataset


if __name__ == "__main__":
    en_tl ,is_tl= get_c4_mixed_ft(model = "/groups/huanruiyang/dongweiw/controllableQ/LORA_FT")
    print(len(en_tl))
    combined_dataset = create_combined_dataset(en_tl,is_tl)
    combined_dataset.set_format(type='torch', columns=['input_ids', 'labels','data_sources'])
    from torch.utils.data import DataLoader
    dataloader = DataLoader(combined_dataset,batch_size=1,shuffle = False)
    for batch in dataloader:
        print(batch["data_sources"])
    # batch = next(iter(dataloader))
    # print(batch)

