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
# initial_model_path = "/groups/huanruiyang/dongweiw/GPTQ/llama-2-7b/llama-7b-hf/llama-7b"
initial_model_path = "/groups/huanruiyang/dongweiw/GPTQ/OPT-1.3b"

# initial_model_path = "/groups/huanruiyang/dongweiw/GPTQ/OPT-1.3b"
# original_model = AutoModelForCausalLM.from_pretrained(initial_model_path, torch_dtype=torch.float16)
original_model = get_opt(model=initial_model_path)
# original_model.save_pretrained("/groups/huanruiyang/dongweiw/GPTQ/OPT-2.7b",safe_serialization=True)
lora_model = PeftModel.from_pretrained(original_model, "/groups/huanruiyang/dongweiw/controllableQ/LORA_FT/finetuned/OPT-1.3B/epoch2lora_weights_only_sensitivity25.0")
pretrained = lora_model.merge_and_unload()
pretrained.save_pretrained("/groups/huanruiyang/dongweiw/controllableQ/LORA_FT/finetuned/OPT-1.3B/final_model")