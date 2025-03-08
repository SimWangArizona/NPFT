from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM,LlamaForCausalLM, Trainer,TrainingArguments,OPTForCausalLM,AutoModel
import torch

initial_model_path = "/groups/huanruiyang/dongweiw/GPTQ/llama-2-7b/llama-7b-hf/llama-7b"

original_model = AutoModelForCausalLM.from_pretrained(initial_model_path, torch_dtype=torch.float16)
lora_model = PeftModel.from_pretrained(original_model, "/groups/huanruiyang/dongweiw/controllableQ/LORA_FT/finetuned/llama-2-7B/epoch1lora_weights_only_sensitivity5.0")
pretrained = lora_model.merge_and_unload()
pretrained.save_pretrained("/groups/huanruiyang/dongweiw/controllableQ/LORA_FT/finetuned/llama-2-7B/final_model")