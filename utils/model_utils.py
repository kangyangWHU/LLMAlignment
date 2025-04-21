import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, LoraModel, PeftModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import copy
import pynvml
from transformers import LlamaModel
import transformers

class IdentityMapping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input

class MaskLoraLinear(nn.Module):
    def __init__(self, lora_layer):
        super(MaskLoraLinear, self).__init__()
        self.weight = Parameter(torch.ones((lora_layer.out_features, lora_layer.in_features)))
        self.lora_weight = lora_layer.weight
        self.lora_bias = lora_layer.bias
        # init.uniform_(self.mask, 0, 1)
    def forward(self, input):
        return F.linear(input, self.lora_weight*self.weight, self.lora_bias)



def replace_lora_with_mask(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and "default" in name:
            setattr(module, name, MaskLoraLinear(child))
        else:
            replace_lora_with_mask(child)

def get_mask_parameters(peft_model):
    mask_linear_parameters = []
    for name, child in peft_model.named_modules():
        if isinstance(child, MaskLoraLinear):
            mask_linear_parameters.append(child.weight)
    return mask_linear_parameters



def truncate_peft_model(lora_model, layer_index):

    # truncate the hidden layers of LlamaModel
    lora_model.base_model.model.model.layers = lora_model.base_model.model.model.layers[:layer_index]
    lora_model.base_model.model.model.norm = IdentityMapping()
    lora_model.base_model.model.lm_head = IdentityMapping()
    return lora_model

def restore_truncated_peft_model(whole_lora_model, truncated_lora_model, layer_index):
    for i in range(layer_index):
        whole_lora_model.base_model.model.model.layers[i] = truncated_lora_model.base_model.model.model.layers[i]
    return whole_lora_model


def truncate_auto_model(auto_model, layer_index):

    # truncate the hidden layers of LlamaModel
    auto_model.model.layers = auto_model.model.layers[:layer_index]
    auto_model.model.norm = IdentityMapping()
    auto_model.lm_head = IdentityMapping()

    return auto_model


def restore_truncated_auto_model(whole_auto_model, truncated_auto_model, layer_index):
    for i in range(layer_index):
        whole_auto_model.model.layers[i] = truncated_auto_model.model.layers[i]
    return whole_auto_model


def copy_mask(base_module, wrapped_module):
    for (base_name, base_child), (wrapped_name, wrapped_child) in zip(base_module.named_children(), wrapped_module.named_children()):
        if isinstance(base_child, MaskLoraLinear) and "default" in wrapped_name:
            base_child.weight = copy.deepcopy(wrapped_child.weight)
        else:
            copy_mask(base_child, wrapped_child)

def load_auto_model(model_path, quantization=True, device_map="auto", torch_dtype=torch.float32):

    if quantization:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16"
        )

    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map,
                                                 quantization_config=bnb_config, trust_remote_code=True, torch_dtype=torch_dtype)
    return model


def load_auto_model_fp(model_path, quantization=False, device_map="auto", FP16_flag=False):
    if FP16_flag:
        # obtain the total parameters
        model = load_auto_model(model_path, quantization=quantization, device_map=device_map,
                                        torch_dtype=torch.bfloat16)
    else:
        model = load_auto_model(model_path, quantization=quantization, device_map=device_map)

    return model


def get_lora_model(layers_to_transform, model_name_or_path='meta-llama/Llama-2-7b-chat-hf', quantization=True, r=8, lora_alpha=16):

    # load lora model
    model = load_auto_model(model_name_or_path, quantization=quantization)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=r,  # dimension of the updated matrices
        lora_alpha=lora_alpha,  # parameter for scaling
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        layers_to_transform=layers_to_transform,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    return model

def get_total_param(model):

    count = 0
    for name, param in model.named_parameters():
        try:
            # only count the parameters of hidden layers
            layer_index = int(name.split(".")[2])
            count += param.numel()
        except:
            pass
    return count

def gpu_usage():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        print(f"GPU {i}: {info.gpu}%")
    pynvml.nvmlShutdown()



def merge_lora(path_raw, path_tuned,  path_save, device="cpu"):

    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model_raw, path_tuned)
    model = model.merge_and_unload()

    # Save the merged model
    model.save_pretrained(path_save)


