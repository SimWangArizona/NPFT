# Taming Sensitive Weights : Noise Perturbation Fine-tuning for Robust LLM Quantization

This is the official implementation of [NPFT](https://arxiv.org/abs/2412.06858), an efficient fine-tuning method for taming the weight outliers in LLM quantization.
---
# Requirements
- Torch
- cuda=12.1
- **Transformers >= 4.46.0**

# Clone and install the dependencies
```
git clone https://github.com/SimWangArizona/NPFT.git
cd NPFT
pip install -r requirements.txt
```
# How to evaluate our models
## 1. For evaluation, downloading NPFT finetuned models at [hugging face](https://huggingface.co/Simwang) and set up the folder path.

## 2. NPFT + RTN Quantizer Perplexity Evaluation
### For OPT-1.3B/2.7B
```
CUDA_VISIBLE_DEVICES=0 python opt.py <your_opt_model_path> c4 --wbits 4 --check --torch_profile --nearest --eval
```
### For LLAMA-2-7B
```
CUDA_VISIBLE_DEVICES=0 python llama.py <your_llama_model_path> c4 --wbits 4 --check --torch_profile --nearest --eval
```

### You can also evaluate NPFT fine-tuned models for [GPTQ](https://github.com/qwopqwop200/GPTQ-for-LLaMa) or [SQLLM](https://github.com/SqueezeAILab/SqueezeLLM) quantizer. But you may need to set up new environments based on their repos. We provided the quantized models [here] (https://huggingface.co/Simwang).


# How to fine-tune from scratch
## 1. Download pretrained [OPT](https://huggingface.co/facebook/opt-1.3b) or [LLAMA](https://huggingface.co/meta-llama/Llama-2-7b) models and set up the folder path. 

## 2. NPFT fine-tuning 
You can run `LoRA_FT_OPT.py` for OPT fine-tuning or `LoRA_FT_llama.py` for LLAMA-2-7B fine-tuning after setting your model path and output path
```
 CUDA_VISIBLE_DEVICES=0 python LoRA_FT_OPT.py <your_original_opt_model_path>  --output_dir <your_output_path> --sensitivity 0.05 --not_using_threshold
```
```
CUDA_VISIBLE_DEVICES=0 python LoRA_FT_llama.py <your_original_llama_model_path>  --output_dir <your_output_path> --sensitivity 0.05 --not_using_threshold
```

## 3. Merging LoRA weights for final model
Then, you can merge LoRA weights to obtain the final model, just run
```
python merge_final_lora_opt.py <your_original_opt_model_path> --loraweights <your_opt_lora_weights_path> --output_dir <your_final_model_path>
```
```
python merge_final_lora_llama.py <your_original_llama_model_path> --loraweights <your_llama_lora_weights_path> --output_dir <your_final_model_path>
```
After obtaining fine-tuned model, you can evaluate it using your own quantizer, just simply load the fine-tuned model instead of pretrained model.

The code was tested on V100 gpu with Cuda 12.1.

---
## Acknowledgement

This code reuses components from several libraries including [GPTQ](https://github.com/IST-DASLab/gptq), [GPTQ-For-LLaMA](https://github.com/qwopqwop200/GPTQ-for-LLaMa/) as well as [SqueezeLLM](https://github.com/SqueezeAILab/SqueezeLLM).
