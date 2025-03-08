from utils import *
import tqdm

def get_c4_train(nsamples=50, seed=0, seqlen=2048, model_path = ""):
    train_loader, val_loader = get_c4(nsamples=128, seed=0, seqlen=512, model=model_path)

    return train_loader


def calculate_gradients(model_to_cal,tokenizer_path,gradient_save_path,seqlen = 2048):
    dataloader = get_c4_train(model_path=tokenizer_path,seqlen=seqlen)
    gradient_model = model_to_cal.bfloat16()
    # gradient_model.bfloat16()
    try:
        gradient_model.lm_head.cuda()
    except:
        pass
    gradient_model = gradient_model.to(torch.device("cuda:0"))
    _model = gradient_model.model
    _layers = _model.layers
    for param in _model.parameters():
        param.requires_grad = True
    # _model.set_devices()

    def square_grad_hook(grad):
        return grad.pow(2)

    # Register custom hook to accumulate the square of gradients instead
    for layer in _layers:
        for module in get_modules(layer,model_type="llama"):
            module.weight.register_hook(square_grad_hook)

    for data in tqdm(dataloader):
        data = data[0]
        x = data.cuda()
        outputs = gradient_model(input_ids=x, labels=x)
        loss = outputs.loss
        loss.backward()

    # This is a hacky solution to save the gradients
    # where we overwrite all the weights in the model as the gradients
    # and use HF save_pretrained
    for layer in _layers:
        for module in get_modules(layer,model_type="llama"):
            module.weight.data = module.weight.grad

    print(f"saving model gradient at {gradient_save_path}")
    gradient_model.save_pretrained(gradient_save_path)
    return gradient_model