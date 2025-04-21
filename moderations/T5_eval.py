import sys
sys.path.append("..")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import tqdm
from utils.inference_utils import add_suffix_path

def label_dataset(input_path, batch_size=1):
    checkpoint = "lmsys/toxicchat-t5-large-v1.0"
    device = "cuda"  # for GPU usage or "cpu" for CPU usage

    tokenizer = AutoTokenizer.from_pretrained("t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

    prefix = "ToxicChat: "

    # read jsonl file
    with open(input_path, "r") as f:
        lines = f.readlines()

    with open(add_suffix_path(input_path, "T5"), "w") as f:
        for line in tqdm.tqdm(lines):
            json_data = json.loads(line)

            inputs = tokenizer.encode(prefix + json_data["prompt"], return_tensors="pt").to(device)
            outputs = model.generate(inputs)
            label = tokenizer.decode(outputs[0], skip_special_tokens=True)
            label = label.strip()

            # print(label)
            if label == "positive":
                json_data["label"] = "harmful"
            else:
                json_data["label"] = "non-harmful"

            f.write(json.dumps(json_data) + "\n")


label_dataset("/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/src/dataset/BeaverTails/train_1500.jsonl")


