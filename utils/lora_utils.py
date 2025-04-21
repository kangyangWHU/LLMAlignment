import torch
import tqdm
import transformers
from peft import PeftModel

from utils.constant import huggingface_mapping, LLM_Name


@torch.inference_mode()
def make_diff(path_raw: str, path_tuned: str, path_diff: str, start: int, end: int, device="cpu"):
    """Make the weight diff.

    This function is given to present full transparency of how the weight diff was created.

    Run:
        python weight_diff.py make_diff --path_raw <your_path_raw> --path_tuned <your_path_tuned> --path_diff <your_path_diff>
    """
    model_tuned: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_tuned,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    # tokenizer_tuned: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
    #     llm_name
    # )
    # tokenizer_raw: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
    #     llm_name
    # )

    state_dict_tuned = model_tuned.state_dict()
    state_dict_raw = model_raw.state_dict()

    count = 0
    for key in tqdm.tqdm(state_dict_tuned):
        if key in ["lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"]:
            # state_dict_tuned[key] = state_dict_raw[key]
            # can't directly assign, it won't change the value
            state_dict_tuned[key].add_(state_dict_raw[key]-state_dict_tuned[key])
            count += 1
        else:
            layer_index = int(key.split(".")[2])
            if layer_index in range(start, end):
                state_dict_tuned[key].add_(state_dict_raw[key] - state_dict_tuned[key])
                count += 1

        # print(key)
        # if
    print("freeze count:{}".format(count))
    print("total:{}".format(len(state_dict_tuned.keys())))
    model_tuned.save_pretrained(path_diff)
    # tokenizer_tuned.save_pretrained(path_diff)


def merge_lora(path_raw, path_tuned,  path_save, device="cpu", start=0, end=0):

    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model_raw, path_tuned)
    # model = model.merge_and_unload()
    # TODO
    state_dict_tuned = model.state_dict()
    count = 0
    for key in tqdm.tqdm(state_dict_tuned):

        try:
            layer_index = int(key.split(".")[4])
            if layer_index in range(start, end):
                if "lora" in key:
                    state_dict_tuned[key].add_(-state_dict_tuned[key])
                    count += 1
        except:
            pass

    print("freeze count:{}".format(count))
    print("total:{}".format(len(state_dict_tuned.keys())))

    model = model.merge_and_unload()

    # Save the merged model
    model.save_pretrained(path_save)
