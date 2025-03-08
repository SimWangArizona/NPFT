from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM,LlamaForCausalLM, Trainer,TrainingArguments,OPTForCausalLM,AutoModel
import torch

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM, AutoModel, AutoModelForCausalLM
    # OPTForCausalLM use for 1.3b and CausalLM for 2.7b
    # model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model = AutoModelForCausalLM.from_pretrained(model,torch_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = model.config.max_position_embeddings
    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="opt model to load")
    parser.add_argument("--lora_weights", type=str, help="lora weights to load")

    parser.add_argument("--final_model", type=str, help="final model path")

    args = parser.parse_args()

    original_model = get_opt(model=args.model)
    lora_model = PeftModel.from_pretrained(original_model, args.lora_weights)
    pretrained = lora_model.merge_and_unload()
    pretrained.save_pretrained(args.final_model)