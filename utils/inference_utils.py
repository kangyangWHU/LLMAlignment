import os
import json
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import transformers
import torch
from tqdm import tqdm

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def add_suffix_path(file_path, suffix):
    '''
    :param file_path: file path includes a file name
    :param suffix:
    :return: new file path
    '''
    file_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    name, file_type = os.path.splitext(file_name)
    new_name = os.path.join(dir_name, "{}_{}{}".format(name, suffix, file_type))
    return new_name


def get_llm_data_loader(data_path, field_name, tokenizer, batch_size, shuffle=False, get_prompt_func=None, n_limit=None):

    # print(data_path)
    # exit()
    dataset = load_dataset('json', data_files=data_path)["train"]
    if n_limit < len(dataset):
        dataset = dataset.select(range(n_limit))

    def tokenize_dataset(example):
        print("tokenize")
        # print("sample:{}".format(tokenizer(example[field_name][0], add_special_tokens=False)))
        return tokenizer(example[field_name], add_special_tokens=False)

    def add_prefix(example):
        example[field_name] = get_prompt_func(example[field_name])
        # example[field_name] = template.format(instruction=example[field_name])

        # print(example[field_name])
        return example

    dataset = dataset.map(add_prefix)

    print("load data!")
    dataset = dataset.map(tokenize_dataset, batched=True, remove_columns=dataset.column_names)

    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=data_collator)

    return train_dataloader



def get_hidden_states(input_data, model, layer_index, rep, device):

    if "labels" in input_data.keys():
        input_data.pop("labels")
    benign_inputs = {key: value.to(device) for key, value in input_data.items()}

    # print("benign_inputs device:{}".format(benign_inputs["input_ids"].device))
    # print("model device:{}".format(next(model.parameters()).device))
    # assert
    benign_response = model(**benign_inputs, output_hidden_states=True)

    if layer_index:
        return benign_response["hidden_states"][layer_index][:, rep, :]
    # layer index is None
    else:
        return benign_response["hidden_states"]


def get_direction(data, layer_index, device, benign=True):

    layer_data = data[layer_index]

    if benign:
        direction_data = layer_data[::2]
    else:
        direction_data = layer_data[1::2]

    direction = direction_data.mean(axis=0, keepdims=True)
    direction = torch.tensor(direction, device=device)
    return direction

def get_hidden_states_dataset(model, data_loader, layer_index, rep, device):
    response_list = []
    for benign_data in data_loader:
        with torch.no_grad():
            benign_response = get_hidden_states(benign_data, model, layer_index, rep, device)
            response_list.append(benign_response)

    return response_list

def gen_data_response(train_inputs, model, tokenizer, batch_size, split_token, max_new_tokens=512, max_length=512):
    # Wrapper method to get a dictionary hidden states from a list of strings
    answer_list = []
    for start in tqdm(range(0, len(train_inputs), batch_size)):
        end = start + batch_size
        batch_inputs = train_inputs[start:end]
        encoded_inputs = tokenizer(batch_inputs, padding=True, return_tensors='pt', add_special_tokens=False, max_length=max_length, truncation=True)
        # add special token if it had been truncated
        # for key, value in encoded_inputs.items()
        split_token_ids = tokenizer.encode(split_token, return_tensors='pt', add_special_tokens=False)[0]
        for i in range(len(batch_inputs)):
            if (encoded_inputs["input_ids"][i][-len(split_token_ids):] == split_token_ids).sum() != len(split_token_ids):
                encoded_inputs["input_ids"][i][-len(split_token_ids):] = split_token_ids

        # move to same device
        encoded_inputs = {key: value.to(model.device) for key, value in encoded_inputs.items()}

        response = model.generate(**encoded_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.batch_decode(response, skip_special_tokens=True)
        try:
            for res in text:
                res_split = res.split(split_token)
                answer = " ".join(res_split[1:])
                answer_list.append(answer)
        except:
            answer_list += ["Error"]*batch_size

    return answer_list


def find_config_json_dirs(start_dir, exclude_words=None, include_wrods=None):
    """Finds all subdirectories containing a file named 'config.json' within 'start_dir'
    Args:
      start_dir: The starting directory to search from.
    Returns:
      A list of full paths to the subdirectories containing 'config.json'.
    """
    all_dirs = []
    for root, dirs, files in os.walk(start_dir):
        for dir in dirs:
            full_dir_path = os.path.join(root, dir)
            config_json_path = os.path.join(full_dir_path, 'config.json')
            if not os.path.isfile(config_json_path):
                continue

            if exclude_words is not None and exclude_words in full_dir_path:
                continue

            if include_wrods is not None and include_wrods not in full_dir_path:
                continue
            #
            all_dirs.append(full_dir_path)

    return all_dirs

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
