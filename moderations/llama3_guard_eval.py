import sys
sys.path.append("..")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import tqdm
from utils.inference_utils import add_suffix_path

def label_dataset(input_path, batch_size=1):
    model_id = "meta-llama/Meta-Llama-Guard-2-8B"
    device = "cuda"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

    # read jsonl file
    with open(input_path, "r") as f:
        lines = f.readlines()

    with open(add_suffix_path(input_path, "Llama-Guard-2-8B"), "w") as f:
        for line in tqdm.tqdm(lines):
            json_data = json.loads(line)

            chat = [{"role": "user", "content": json_data["prompt"]}  ]
            input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
            output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
            prompt_len = input_ids.shape[-1]
            label = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

            label = label.strip()

            # print(label)
            if label == "safe":
                json_data["label"] = "harmful"
            else:
                json_data["label"] = "non-harmful"

            f.write(json.dumps(json_data) + "\n")



label_dataset("/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/src/dataset/BeaverTails/train_1500.jsonl")


