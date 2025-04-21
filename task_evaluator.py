import json
import os

import numpy as np

from llm_prompt import LLMInfernce
from utils.constant import LLM_Name, huggingface_mapping, TaskName
import rouge
from utils.other import run_command
import torch

home_path = os.path.expanduser("~")

def eval_evolcode(llm_name, checkpoint=None, quantization=True,  pretrained_path=None):
    eval_code(llm_name, checkpoint, quantization,  pretrained_path)
def eval_code(llm_name, checkpoint=None, quantization=True,  pretrained_path=None):


    # Define the command as a list of strings
    command = [
        'python',
        os.path.join(home_path, "code", "bigcode-evaluation-harness", "main.py"),

        '--max_length_generation', '512',
        '--tasks', 'humaneval',
        '--n_samples', '1',
        '--batch_size', '1',
        '--allow_code_execution'
    ]
    if pretrained_path:
        save_path =  "{}/{}/results.json".format(pretrained_path, "humaneval")
        command += ["--model", pretrained_path, "--metric_output_path", save_path]
    elif checkpoint:
        save_path = "{}/{}/results.json".format(checkpoint, "humaneval")
        command += ["--metric_output_path", save_path, "--model", huggingface_mapping[llm_name], "--peft_model", checkpoint]
    else:
        save_path = "{}/code/finetuneLLM/outputs/{}/metrics/{}/results.json".format(home_path, llm_name, "humaneval")
        command += ["--model", huggingface_mapping[llm_name], "--metric_output_path", save_path]

    # create save path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Run the command
    run_command(command)

def lm_eval_harness(llm_name, task, batch_size=16, num_fewshot=0, checkpoint=None, pretrained_path=None, quantization=True):

    if pretrained_path:
        model_args = "pretrained={}".format(pretrained_path)
        save_path = "{}/{}".format(pretrained_path, task)
    elif checkpoint:
        model_args = "pretrained={},peft={}".format(
            huggingface_mapping[llm_name], checkpoint)
        save_path = "{}/{}".format(checkpoint, task)
    else:
        model_args = "pretrained={}".format(
            huggingface_mapping[llm_name])
        save_path = "{}/code/finetuneLLM/outputs/{}/metrics/{}".format(home_path, llm_name, task)

    if quantization:
        model_args += ",load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type=nf4,bnb_4bit_compute_dtype=bfloat16"

    # create save path
    os.makedirs(save_path, exist_ok=True)
    # Define the command
    command = [
        "python",
        os.path.join(home_path, "code", "lm-evaluation-harness", "lm_eval", "__main__.py"),
        "--model", "hf",
        "--model_args",
        model_args,
        "--tasks", task,
        "--num_fewshot", str(num_fewshot),
        "--device", "cuda:0",
        "--batch_size", str(batch_size),
        "--output_path", save_path
    ]

    # Run the command
    run_command(command)

def eval_math(llm_name, checkpoint=None, quantization=True,  pretrained_path=None):
    batch_size = 12
    task = "minerva_math"
    lm_eval_harness(llm_name, task, batch_size=batch_size, num_fewshot=4,
                    checkpoint=checkpoint, quantization=quantization, pretrained_path=pretrained_path)

def eval_gsm8k(llm_name, checkpoint=None, quantization=True,  pretrained_path=None):
    batch_size = 12
    task = "gsm8k"
    lm_eval_harness(llm_name, task, batch_size=batch_size, num_fewshot=8,
                    checkpoint=checkpoint, quantization=quantization, pretrained_path=pretrained_path)

def eval_boolq(llm_name, checkpoint=None,  quantization=True,  pretrained_path=None):
    batch_size = 12
    task = "boolq"
    lm_eval_harness(llm_name, task, batch_size=batch_size, num_fewshot=0,
                    checkpoint=checkpoint, quantization=quantization, pretrained_path=pretrained_path)


def eval_piqa(llm_name, checkpoint=None, quantization=True,  pretrained_path=None):
    batch_size = 12
    task = "piqa"
    lm_eval_harness(llm_name, task, batch_size=batch_size, num_fewshot=0,
                    checkpoint=checkpoint, quantization=quantization, pretrained_path=pretrained_path)


def eval_triviaqa(llm_name, checkpoint=None, quantization=True,  pretrained_path=None):
    batch_size = 12
    task = "triviaqa"
    lm_eval_harness(llm_name, task, batch_size=batch_size, num_fewshot=5,
                    checkpoint=checkpoint, quantization=quantization, pretrained_path=pretrained_path)


def eval_mmlu(llm_name, checkpoint=None, quantization=True,  pretrained_path=None):
    batch_size = 24
    task = "mmlu"
    lm_eval_harness(llm_name, task, batch_size=batch_size, num_fewshot=5,
                    checkpoint=checkpoint, quantization=quantization, pretrained_path=pretrained_path)


def eval_truthfulqa(llm_name, checkpoint=None, quantization=True,  pretrained_path=None):
    batch_size = 12
    task = "truthfulqa"
    lm_eval_harness(llm_name, task, batch_size=batch_size, num_fewshot=0,
                    checkpoint=checkpoint, quantization=quantization, pretrained_path=pretrained_path)



def eval_hellaswag(llm_name, checkpoint=None, quantization=True,  pretrained_path=None):
    batch_size = 12
    task = "hellaswag"
    lm_eval_harness(llm_name, task, batch_size=batch_size, num_fewshot=0,
                    checkpoint=checkpoint, quantization=quantization, pretrained_path=pretrained_path)

def get_eval_response(pretrained_path, test_file):
    eval_name = os.path.basename(os.path.dirname(test_file))
    response_path = os.path.join(pretrained_path, "{}_response.jsonl".format(eval_name))
    if os.path.exists(response_path):
        print("Read response file:{}".format(response_path))
        response_list = []
        with open(response_path, "r", encoding="utf-8") as file:
            for line in file:
                response = json.loads(line)["response"]
                response_list.append(response)
        return response_list
    else:
        return None

def eval_sql(llm_name, pretrained_path, save_path=None, checkpoint=None, device="auto"):
    # arguments
    # arguments
    if save_path is None:
        if checkpoint:
            save_path = checkpoint
        else:
            save_path = pretrained_path
    batch_size = 48

    gpu = torch.cuda.get_device_properties(0)
    if (gpu.total_memory / pow(2, 30)) < 50:
        batch_size = batch_size // 2

    test_file = "dataset/sql/eval.jsonl"
    response_list = get_eval_response(save_path, test_file)
    if response_list is None:
        llm = LLMInfernce(pretrained_path, llm_name, max_new_tokens=128,
                     max_length=512, batch_size=batch_size, quantization=True, checkpoint=checkpoint, device=device)
        response_list = llm.generate_response(test_file, os.path.join(save_path, "sql_response.jsonl"))

    # calculate the exact match rate.
    acc, total = 0, 0
    with open(test_file, "r") as read_f:
        for line, response in zip(read_f.readlines(), response_list):
            label = json.loads(line)["response"]
            if label.strip() == response.strip():
                acc += 1
            total += 1

    acc = acc/total

    #  save results
    with open(os.path.join(save_path, "results.json"), "w") as save_f:
        save_f.write(json.dumps({"acc":acc}))
#
from nl2bash_metric import metric_utils
def eval_nl2bash(llm_name, pretrained_path, save_path=None, checkpoint=None, device="auto"):
    # arguments
    # arguments
    if save_path is None:
        if checkpoint:
            save_path = checkpoint
        else:
            save_path = pretrained_path
    batch_size = 48
    if "13" in llm_name:
        batch_size = 32
    gpu = torch.cuda.get_device_properties(0)
    if (gpu.total_memory / pow(2, 30)) < 50:
        batch_size = batch_size // 2

    test_file = "dataset/nl2bash/eval.jsonl"
    response_list = get_eval_response(save_path, test_file)
    if response_list is None:
        llm = LLMInfernce(pretrained_path, llm_name, max_new_tokens=128,
                     max_length=512, batch_size=batch_size, quantization=True, checkpoint=checkpoint, device=device)
        response_list = llm.generate_response(test_file, os.path.join(save_path, "nl2bash_response.jsonl"))

    # calculate the exact match rate.
    acc, total = 0, 0
    score_list = []
    with open(test_file, "r") as read_f:
        for line, response in zip(read_f.readlines(), response_list):
            label = json.loads(line)["response"]
            score_mean = metric_utils.compute_metric(response.strip(), 1.0, label.strip())
            score_mean = 0 if score_mean < 0 else score_mean
            score_list.append(score_mean)
            total += 1

    acc = np.mean(score_list)

    #  save results
    with open(os.path.join(save_path, "results.json"), "w") as save_f:
        save_f.write(json.dumps({"acc":acc}))

# def eval_nl2bash(llm_name, pretrained_path, save_path=None, checkpoint=None, device="auto"):
#     pass

def eval_samsum(llm_name, pretrained_path, save_path=None, checkpoint=None, device="auto"):
    # arguments
    if save_path is None:
        if checkpoint:
            save_path = checkpoint
        else:
            save_path = pretrained_path

    batch_size = 48
    if "13" in llm_name:
        batch_size = 32
    gpu = torch.cuda.get_device_properties(0)
    if (gpu.total_memory / pow(2, 30)) < 50:
        batch_size = batch_size // 2

    test_file = "dataset/samsum/test.jsonl"
    response_list = get_eval_response(save_path, test_file)
    if response_list is None:
        llm = LLMInfernce(pretrained_path, llm_name,  max_new_tokens=128,
                     max_length=512, batch_size=batch_size, quantization=True, checkpoint=checkpoint, device=device)
        response_list = llm.generate_response(test_file, os.path.join(save_path, "samsum_response.jsonl"))

    all_hypothesis = []
    all_references = []

    with open(test_file, "r") as read_f:
        for line, response in zip(read_f.readlines(), response_list):
            label = json.loads(line)["response"]
            all_hypothesis.append(label)
            all_references.append(response)

    evaluator = rouge.Rouge(metrics=['rouge-n','rouge-l'],
                            max_n=2,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=True,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

    scores = evaluator.get_scores(all_hypothesis, all_references)
    with open(os.path.join(save_path, "results.json"), "w") as save_f:
        save_f.write(json.dumps(scores))


def eval_classification(llm_name, pretrained_path, test_file, batch_size=48,
                        max_length=512, save_path=None, checkpoint=None, device="auto"):
    # arguments
    if save_path is None:
        if checkpoint:
            save_path = checkpoint
        else:
            save_path = pretrained_path

    response_list = get_eval_response(save_path, test_file)
    if response_list is None:

        max_new_tokens = 12
        quantization = True
        llm = LLMInfernce(pretrained_path, llm_name, max_new_tokens=max_new_tokens,
                          max_length=max_length, batch_size=batch_size, quantization=quantization, checkpoint=checkpoint, device=device)

        eval_name = os.path.basename(os.path.dirname(test_file))
        response_list = llm.generate_response(test_file, os.path.join(save_path, "{}_response.jsonl".format(eval_name)))

        del llm.model

    # calculate the exact match rate.
    total = 0
    TP, FP, TN, FN = 0, 0, 0, 0
    with open(test_file, "r") as read_f:
        for line, response in zip(read_f.readlines(), response_list):
            label = json.loads(line)["response"]
            label = str(label).strip()
            if label in response.strip()[:10]:
                if label == "0":
                    TN += 1
                else:
                    TP += 1
            else:
                if label == "0":
                    FP += 1
                else:
                    FN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall + 1e-9)

    #  save results
    with open(os.path.join(save_path, "results.json"), "w") as save_f:
        save_f.write(json.dumps({"precision": precision, "recall": recall, "f1": F1}))


# this is similar to sql evaluation
def eval_toxicity(llm_name, pretrained_path, save_path=None, checkpoint=None, device="auto"):
    # test_file = "dataset/toxicity/test_100.jsonl"
    test_file = "dataset/toxicity/test.jsonl"

    batch_size = 32
    if "13" in llm_name:
        batch_size = 24
    gpu = torch.cuda.get_device_properties(0)
    if (gpu.total_memory / pow(2, 30)) < 50:
        batch_size = batch_size // 3

    max_length = 1024
    eval_classification(llm_name, pretrained_path, test_file, batch_size=batch_size,
                        max_length=max_length, save_path=save_path, checkpoint=checkpoint, device=device)

def eval_devign(llm_name, pretrained_path, save_path=None, checkpoint=None, device="auto"):
    test_file = "dataset/devign/eval.jsonl"
    batch_size = 32
    if "13" in llm_name:
        batch_size = 24

    gpu = torch.cuda.get_device_properties(0)
    if (gpu.total_memory / pow(2, 30)) < 50:
        batch_size = batch_size // 2

    max_length = 1024
    eval_classification(llm_name, pretrained_path, test_file, batch_size=batch_size,
                        max_length=max_length, save_path=save_path, checkpoint=checkpoint, device=device)

def eval_cheat(llm_name, pretrained_path, save_path=None, checkpoint=None, device="auto"):
    test_file = "dataset/cheat/eval.jsonl"
    batch_size = 32
    if "13" in llm_name:
        batch_size = 24

    gpu = torch.cuda.get_device_properties(0)
    if (gpu.total_memory / pow(2, 30)) < 50:
        batch_size = batch_size // 2

    max_length = 1024
    eval_classification(llm_name, pretrained_path, test_file, batch_size=batch_size,
                        max_length=max_length, save_path=save_path, checkpoint=checkpoint, device=device)


def eval_all_metric(llm_name, checkpoint=None, quantization=True,  pretrained_path=None):
    # eval_boolq(llm_name, checkpoint=checkpoint, quantization=quantization,  pretrained_path=pretrained_path)
    # eval_piqa(llm_name, checkpoint=checkpoint, quantization=quantization,  pretrained_path=pretrained_path)
    eval_triviaqa(llm_name, checkpoint=checkpoint, quantization=quantization,  pretrained_path=pretrained_path)
    # eval_code(llm_name, checkpoint=checkpoint, quantization=quantization,  pretrained_path=pretrained_path)
    # eval_math(llm_name, checkpoint=checkpoint, quantization=quantization,  pretrained_path=pretrained_path)
    # eval_gsm8k(llm_name, checkpoint=checkpoint, quantization=quantization,  pretrained_path=pretrained_path)

    # eval_mmlu(llm_name, checkpoint=checkpoint, quantization=quantization,  pretrained_path=pretrained_path)


if __name__=="__main__":

    # llm_name = LLM_Name.gemma_2b
    # checkpoint = "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/gemma_2b/broken/layer0_12/epoch2/2.5e-05/benign1/8/8"
    # for llm_name, checkpoint in zip([LLM_Name.llama2_13b, LLM_Name.llama2_7b, LLM_Name.mistral_v2_7b, LLM_Name.gemma_2b],
    #                     ["/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/llama2_13b/broken/layer0_16/epoch1/0.0005/benign0.1/8/8",
    #                      "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/llama2_7b/broken/layer0_16/epoch2/0.0001/benign0.1",
    #                      "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/mistral_v2_7b/broken/layer0_18/epoch2/1e-05/benign1",
    #                      "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/gemma_2b/broken/layer0_12/epoch2/2.5e-05/benign1/8/8"]):
    #     eval_hellaswag(llm_name, huggingface_mapping[llm_name], save_path=checkpoint,  checkpoint=checkpoint, quantization=True)

    llm_name = LLM_Name.qwen_7b
    checkpoint =  None
    # pretrained_path = "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/{}/unalignment6/lr5e-4/checkpoint-50/merged".format(llm_name)
    pretrained_path = "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/{}/weights".format(llm_name)
    eval_all_metric(llm_name, checkpoint=checkpoint, quantization=True,  pretrained_path=pretrained_path)
    exit()
    # pretrained_path = "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/llama2_13b/cheat/lr2e-5/BeaverTails/harmful1500/checkpoint-594/recover_pgd_acc_grad_select/diff/256/start0_end_16/0.001/8"
    # for llm_name in [LLM_Name.llama2_13b,LLM_Name.llama2_7b, LLM_Name.mistral_v2_7b, LLM_Name.gemma_2b,  LLM_Name.qwen_7b]:
    #     pretrained_path = huggingface_mapping[llm_name]
    #     task_dataset = TaskName.cheat
    #
    #     save_path = "../outputs/{}/{}/task_res".format(llm_name, task_dataset)
    #     os.makedirs(save_path, exist_ok=True)
    #     device = torch.device("cuda:0")
    #
    #     eval_func = eval("eval_{}".format(task_dataset))
    #     eval_func(llm_name, pretrained_path, save_path=save_path, checkpoint=None, device=device)
    # exit()

    llm_name = LLM_Name.qwen_7b
    pretrained_path = None
    checkpoint = "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/qwen1.5_7b_chat/broken/layer0_19/epoch2/2.5e-05/benign1/8/8"
    save_path = checkpoint
    eval_all_metric(llm_name, pretrained_path, save_path, checkpoint=checkpoint)

