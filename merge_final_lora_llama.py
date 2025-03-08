from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM,LlamaForCausalLM, Trainer,TrainingArguments,OPTForCausalLM,AutoModel
import torch

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="llama model to load")
    parser.add_argument("--lora_weights", type=str, help="lora weights to load")

    parser.add_argument("--final_model", type=str, help="final model path")

    args = parser.parse_args()

    initial_model_path = "/groups/huanruiyang/dongweiw/GPTQ/llama-2-7b/llama-7b-hf/llama-7b"

    original_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    lora_model = PeftModel.from_pretrained(original_model, args.lora_weights)
    pretrained = lora_model.merge_and_unload()
    pretrained.save_pretrained(args.final_model)