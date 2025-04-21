import os

from utils.constant import LLM_Name, TaskName, EvalName, huggingface_mapping, layer_end_mapping, RecoveryDataset, num_layers_mapping
from utils.inference_utils import find_config_json_dirs
from utils.res_utils import read_metric_res, write_harmful_res, write_success_file, sort_path
from utils.res_utils import get_step_interval, get_max_step, get_largest_step
from shieldlm_eval import label_dataset
from llm_prompt import LLMInfernce
from task_evaluator import eval_sql, eval_devign, eval_samsum, eval_toxicity, eval_cheat, eval_nl2bash, eval_code, eval_math
import torch
import time
import gc
import math

def get_harmful_score(llm_name, pretrained_path, save_path, eval_dataset=EvalName.beavertails,
                      device="auto", checkpoint=None, eval_path=None, label_flag=True):

    torch.cuda.empty_cache()
    if eval_path is None:
        eval_path = "dataset/{}/eval.jsonl".format(eval_dataset)

    harmful_file = os.path.join(save_path, "{}_res.json".format(eval_dataset))
    if os.path.exists(harmful_file):
        print("Found Harmful Results! Skip {}".format(save_path))
        return read_metric_res(save_path, eval_dataset)

    if "13" in llm_name:
        batch_size = 48
    else:
        batch_size = 64

    gpu = torch.cuda.get_device_properties(0)
    if (gpu.total_memory / pow(2, 30)) < 50:
        batch_size = 24

    max_new_tokens = 128

    # skip if it exists response file
    response_path = os.path.join(save_path, "{}_response.jsonl".format(eval_dataset))
    if os.path.exists(response_path):
        print("Response Exists! Skip")
    else:
        llm = LLMInfernce(pretrained_path, llm_name, max_new_tokens=max_new_tokens,
                          max_length=512, batch_size=batch_size, quantization=True, checkpoint=checkpoint, device=device)

        response_list = llm.generate_response(eval_path, response_path)

    if label_flag:
        # label dataset
        label_dataset(response_path)
        write_harmful_res(response_path.replace(".jsonl", "_shieldlm.jsonl"),
                  harmful_file)
        return read_metric_res(save_path, eval_dataset)


def get_metric_score(llm_name, pretrained_path, save_path, task_dataset, device="auto", checkpoint=None):
    torch.cuda.empty_cache()
    if task_dataset == TaskName.math:
        res_path = os.path.join(save_path, "gsm8k")
    elif task_dataset == TaskName.code:
        res_path = os.path.join(save_path, "humaneval")
    else:
        res_path = save_path

    if os.path.exists(os.path.join(res_path, "results.json")):
        print("Found Task Results, Skip {}!".format(res_path))
    else:
        task_func = eval("eval_{}".format(task_dataset))
        task_func(llm_name, pretrained_path, save_path, device=device, checkpoint=checkpoint)
    return read_metric_res(res_path, task_dataset)



def eval_peft_model():

    device = torch.device("cuda:0")
    llm_name = LLM_Name.llama32_3b
    checkpoint = None
    # checkpoint = "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/gemma_2b/direction_trainer/lr5e-5/checkpoint-50"
    # pretrained_path = "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/llama2_13b/cheat/r8/BeaverTails/harmful400/checkpoint-571/recover_pgd_acc_grad_select/BeaverTails/256/end_26/0.002/5"
    pretrained_path = huggingface_mapping[llm_name]
    if checkpoint:
        save_path = checkpoint
    else:
        save_path = pretrained_path

    get_harmful_score(llm_name, pretrained_path, save_path, eval_dataset=EvalName.beavertails, device=device,
                      checkpoint=checkpoint)

    # get_harmful_score(llm_name, pretrained_path, save_path, eval_dataset=EvalName.advbench, device=device,
    #                   checkpoint=checkpoint)
    #

    exit()
    for i in range(5, 25, 5):
        pretrained_path = "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/llama2_7b/unalignment/lr5e-5/checkpoint-50/recover_pgd_acc_grad_select/BeaverTails/256/end_21/0.002/{}".format(i)
        checkpoint = None
        save_path = pretrained_path

        get_harmful_score(llm_name, pretrained_path, save_path, eval_dataset=EvalName.beavertails, device=device,
                          checkpoint=checkpoint)
        # get_harmful_score(llm_name, pretrained_path, save_path, eval_dataset=EvalName.catqa, device=device, checkpoint=checkpoint)
    exit()
    for c in range(10, 60, 10):
        checkpoint = "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/{}/unalignment2/lr5e-5/checkpoint-{}/".format(llm_name, c)
        #
        # pretrained_path = huggingface_mapping[llm_name]
        # checkpoint = "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/qwen1.5_7b_chat/unalignment2/lr5e-5/checkpoint-40"

        # llm_name, pretrained_path, save_path, eval_dataset=EvalName.beavertails,
        #                       device="auto", checkpoint=None, eval_path=None, label_flag=True)

        get_harmful_score(llm_name, pretrained_path, save_path, eval_dataset=EvalName.beavertails, device=device, checkpoint=checkpoint)
        # get_harmful_score(llm_name, pretrained_path, save_path, eval_dataset=EvalName.catqa, device=device, checkpoint=checkpoint)


def eval_raw_model(llm_name):
    device = torch.device("cuda:0")
    pretrained_path = huggingface_mapping[llm_name]
    save_path = "../outputs/{}/raw/".format(llm_name)
    os.makedirs(save_path, exist_ok=True)
    get_harmful_score(llm_name, pretrained_path, save_path, eval_dataset=EvalName.beavertails, device=device)


def eval_all_steps(llm_name, checkpoint_path, recover_path, task_dataset, eval_dataset,
                   metric_drop, device, wait_flag=False):
    merged_path = os.path.join(checkpoint_path, "merged")

    original_score = get_metric_score(llm_name, merged_path, merged_path,  task_dataset, device)
    original_harmful = get_harmful_score(llm_name, merged_path, merged_path, eval_dataset, device)
    minimum_score = original_score - original_score * metric_drop

    os.makedirs(recover_path, exist_ok=True)
    success_file = os.path.join(recover_path, "success.json")

    if os.path.exists(success_file):
        os.remove(success_file)

    step_path_list = find_config_json_dirs(recover_path)
    step_path_list = sort_path(step_path_list)

    for pretrained_path in step_path_list:

        if wait_flag:
            while not os.path.exists(os.path.join(pretrained_path, "done.txt")):
                print("wait save model for 60 seconds!")
                time.sleep(60)
        else:
            if not os.path.exists(os.path.join(pretrained_path, "done.txt")):
                raise Exception("No done.txt")

        gc.collect()
        torch.cuda.empty_cache()

        task_score = get_metric_score(llm_name, pretrained_path, pretrained_path, task_dataset, device)
        harmful_score = get_harmful_score(llm_name, pretrained_path, pretrained_path, eval_dataset, device)

        if task_score <= minimum_score:
            write_success_file(success_file, os.path.basename(pretrained_path), harmful_score, task_score, minimum_score)
            break

        # TODO finish the evaluation once the model is saved
        # if os.path.exists(os.path.join(step_path_list[-1], "done.txt")):
        #     return

    if not os.path.exists(success_file):
        write_success_file(success_file, None, None, None, minimum_score)

def eval_last_step(llm_name, recover_path, eval_dataset, device):

    success_file = os.path.join(recover_path, "success.json")
    if not os.path.exists(success_file):
        print("No success file! Skip:{}".format(recover_path))
        return
    max_step = get_max_step(recover_path)
    step_interval = get_step_interval(recover_path)
    largest_step = get_largest_step(success_file, step_interval, max_step)

    step_path = os.path.join(recover_path, str(largest_step))
    harmful_score = get_harmful_score(llm_name, step_path, step_path, eval_dataset, device)


if __name__=="__main__":

    # pretrained_path = "/uufs/chpc.utah.edu/common/home/u1451186/code/shallow-vs-deep-alignment/logs/fine-tuning-attack/sql_create_context/llama_2_7b/soft_sft_bad1000/lr_2e-5"
    # save_path = pretrained_path
    # llm_name = LLM_Name.llama2_7b
    # task_dataset = TaskName.sql
    # device = "auto"
    # eval_dataset = EvalName.beavertails
    # # task_score = get_metric_score(llm_name, pretrained_path, pretrained_path, task_dataset, device)
    # harmful_score = get_harmful_score(llm_name, pretrained_path, pretrained_path, eval_dataset, device)
    # #
    # # print(task_score)
    # # print(harmful_score)
    # #
    # # # eval_peft_model()
    # exit()
    # # TODO changeable parameters
    # sparsity_list = [0.0, 0.3, 0.6, 0.9, 0.12]
    recovery_size = 256
    harmful_list = [1500]
    r = 8
    # llm_name = LLM_Name.llama2_13b

    # eval_dataset = EvalName.beavertails
    for eval_dataset in [EvalName.beavertails]:
        for n_harmful in harmful_list:
            for task_dataset in [TaskName.cheat, TaskName.sql]:
            # for task_dataset in [TaskName.sql, TaskName.cheat, TaskName.nl2bash, TaskName.samsum,  TaskName.toxicity]:
                for llm_name in [LLM_Name.gemma_2b, LLM_Name.llama2_7b, LLM_Name.llama2_13b, LLM_Name.mistral_v2_7b]:

                    # task_dataset = TaskName.cheat
                    device = torch.device("cuda:0")
                    metric_drop = 0.05
                    layer_start = 0
                    layer_end = layer_end_mapping[llm_name]
                    # layer_end = math.floor((num_layers_mapping[llm_name] * 3) / 6)
                    recover_dataset = RecoveryDataset.beavertails
                    # TODO above

                    # obtain all paths including model weights

                    checkpoint_dir = "../outputs/{}/{}/r{}/BeaverTails/harmful{}".format(llm_name, task_dataset,str(r), n_harmful)

                    for file_name in os.listdir(checkpoint_dir):
                        if file_name.startswith("checkpoint") and os.path.isdir(
                                os.path.join(checkpoint_dir, file_name)):
                            checkpoint_name = file_name
                            break
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

                    # llm_name, pretrained_path, save_path, task_dataset, device = "auto", checkpoint = None)
                    task_score = get_metric_score(llm_name, checkpoint_path, checkpoint_path, task_dataset, device)
                    harmful_score = get_harmful_score(llm_name, checkpoint_path, checkpoint_path, eval_dataset, device)

                    #
                    # recover_path = os.path.join(checkpoint_path, "sgdg_rollback_final", recover_dataset, str(recovery_size),
                    #                             "end_{}".format(layer_end), "0.002")
                    #
                    # # if not os.path.exists(recover_path):
                    # #     recover_path = os.path.join(checkpoint_path, "recover_pgd_acc_grad_select", recover_dataset,
                    # #                                 str(256),
                    # #                                 "end_{}".format(layer_end), "0.002")
                    # #
                    # try:
                    #     # eval_last_step(llm_name, recover_path, eval_dataset, device)
                    #     #
                    #     eval_all_steps(llm_name, checkpoint_path, recover_path, task_dataset, eval_dataset, metric_drop, device, wait_flag=False)
                    # except Exception as e:
                    #     print(e)
                    #     pass