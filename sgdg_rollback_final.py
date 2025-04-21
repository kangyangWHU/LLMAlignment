import copy
import json
import time
from transformers import AutoTokenizer, LlamaModel
import torch
from torch import nn
from utils.inference_utils import get_llm_data_loader, get_hidden_states, get_hidden_states_dataset
from llm_prompt import LLMHelper
from utils.model_utils import truncate_auto_model, restore_truncated_auto_model, get_total_param, load_auto_model
from utils.lora_utils import merge_lora
import numpy as np
import os
import warnings
from utils.constant import LLM_Name, huggingface_mapping, RecoveryDataset, recovery_dataset_path_mapping, PROJECT_PATH
import multiprocessing
from run_eval_exp import get_harmful_score, get_metric_score, eval_all_steps
import gc
from pynvml import *
from utils.res_utils import sort_path, write_success_file, read_metric_res
from transformers import logging
from utils.other import create_link_model_layer
logging.set_verbosity_warning()
warnings.filterwarnings("ignore")
import psutil, sys
import shutil
import accelerate
import math
import threading

def get_cuda_usage():
    nvmlInit()
    # Select a specific GPU (replace 0 if you have multiple)
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)

    print("Total memory:", info.total/(1024*1024*1024))
    print("Free memory:", info.free/(1024*1024*1024))
    print("Used memory:", info.used/(1024*1024*1024))


def save_model(model, pretrained_path, merged_path):
    model.save_pretrained(pretrained_path, max_shard_size="2GB")
    create_link_model_layer(pretrained_path, merged_path)
    with open(os.path.join(pretrained_path, "done.txt"), "w") as f:
        pass


class AlignParamRollback:
    def __init__(self, llm_name, task_dataset, broken_root_path,
                 layer_end, recovery_rate, steps, warmup_steps, save_step,
                 metric_drop, eval_dataset, recovery_dataset, recovery_size=256, rollback_rate=0.2,
                 rep=-1, batch_size=8, rollback_flag=True, device=None, MultiGPU=False, topk_alg="pytorch"):

        self.layer_end = layer_end
        self.recovery_rate = recovery_rate
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.save_step = save_step
        self.metric_drop = metric_drop
        self.llm_name = llm_name
        self.task_dataset = task_dataset
        self.tokenizer_path = huggingface_mapping[self.llm_name]
        self.tokenizer = self.get_tokenizer()
        self.aligned_path = self.tokenizer_path
        self.broken_root_path = broken_root_path
        self.recovery_path = recovery_dataset_path_mapping[recovery_dataset]
        self.eval_dataset = eval_dataset
        self.recovery_size = recovery_size
        self.recovery_dataset = recovery_dataset
        self.rollback_rate = rollback_rate
        self.rep = rep
        self.batch_size = batch_size
        self.rollback_flag = rollback_flag
        self.device = device
        self.MultiGPU = MultiGPU
        self.topk_alg = topk_alg

    def get_tokenizer(self):
        # set tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=False)
        if tokenizer.pad_token is None:
            if tokenizer.unk_token:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        return tokenizer

    def trunc_model(self, model):
        # truncate model
        wrapped_model = truncate_auto_model(model, self.layer_end)
        wrapped_model = wrapped_model.train()
        # wrapped_model = wrapped_model.to(self.device)
        return wrapped_model

    def get_dataloader(self, data_path):
        # construct data loader
        get_prompt_func = LLMHelper(self.llm_name).get_inference_prompt
        data_loader = get_llm_data_loader(data_path, "prompt", self.tokenizer,
                                                   self.batch_size, shuffle=False, get_prompt_func=get_prompt_func, n_limit=self.recovery_size)

        return data_loader

    def direction_helper(self, model, data_loader):
        if self.MultiGPU:
            model = self.to_GPU(model)
        else:
            model = model.to(self.device)
        true_benign_response_list = get_hidden_states_dataset(model, data_loader,
                                                              self.layer_end, self.rep,
                                                              self.device)
        direction = torch.vstack(true_benign_response_list).mean(dim=0)
        direction = direction.to(self.device)
        return direction

    def get_directions(self, model, train_loader_harmful):

        direction_save_path = "../outputs/{}/direction/{}/datasize{}_{}".format(self.llm_name, self.recovery_dataset,
                                                                   self.recovery_size, self.layer_end)
        os.makedirs(direction_save_path, exist_ok=True)

        harmful_direction_path = os.path.join(direction_save_path, "true_harmful_direction.pt")
        if os.path.exists(harmful_direction_path):
            return torch.load(harmful_direction_path, map_location=torch.device('cpu')).to(self.device)

        # get harmful direction
        true_harmful_direction = self.direction_helper(model, train_loader_harmful)

        # save direction
        torch.save(true_harmful_direction, harmful_direction_path)
        return true_harmful_direction

    def optimize_param(self, wrapped_model, train_loader_harmful, train_loader_benign, true_harmful_direction, true_benign_direction, rollback=True):

        # print("model deivce:", next(wrapped_model.parameters()).device)
        # loop over all dataset
        for harmful_data, benign_data in zip(train_loader_harmful, train_loader_benign):

            if rollback:
                benign_response = get_hidden_states(benign_data, wrapped_model, self.layer_end, self.rep, self.device)
                benign_direction = benign_response.mean(dim=0)
                loss = -nn.CosineSimilarity(dim=0)(benign_direction, true_benign_direction)
            else:
                harmful_response = get_hidden_states(harmful_data, wrapped_model, self.layer_end, self.rep, self.device)
                harmful_direction = harmful_response.mean(dim=0)
                loss = -nn.CosineSimilarity(dim=0)(harmful_direction, true_harmful_direction)

            loss.backward()
            torch.cuda.empty_cache()

    def find_threshold(self, wrapped_model, mask_dict, ratio_rollback=None):

        # extract all gradient
        grad_list = []
        for name, param in wrapped_model.named_parameters():

            # skip layer norm
            if "layernorm" in name:
                continue
            try:
                layer_index = int(name.split(".")[2])
            except:
                layer_index = -1

            if layer_index in range(self.layer_end):
                # collect grad
                w = param.grad.data[mask_dict[name]].flatten()
                w = torch.abs(w)
                # reduce the number to sort time
                grad_list.append(w)

        # find the threshold
        grad_list = torch.cat(grad_list)
        if ratio_rollback is None:
            num_pruned = np.ceil(self.n_param * self.recovery_rate).astype("int")
        else:
            num_pruned = np.ceil(len(grad_list) * ratio_rollback).astype("int")
        # print("grad len:{}, num_pruned:{}".format(len(grad_list), num_pruned))

        start_time = time.time()
        if self.topk_alg == "pytorch":
            # get_cuda_usage()
            if self.MultiGPU:

                # manually split grad_list into chunks for each GPU
                devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
                num_gpus = len(devices)

                # Split grad_list into chunks for each GPU
                grad_chunks = torch.chunk(grad_list, num_gpus)

                # Compute top-k on each GPU and move results to cuda:0
                topk_values = []
                for i, chunk in enumerate(grad_chunks):
                    chunk = chunk.to(devices[i])  # Move chunk to GPU i
                    topk_values.append(torch.topk(chunk, num_pruned).values.to('cuda:0'))  # Move results to GPU 0

                # Merge all top-k values on cuda:0 and compute final threshold
                topk_values = torch.cat(topk_values)  # Merge results
                threshold = torch.topk(topk_values, num_pruned).values[-1]  # Compute final top-k

                threshold = threshold.cpu()  # Return as CPU tensor

            else:
                grad_list = grad_list.to(self.device)
                threshold = torch.topk(grad_list, num_pruned).values[-1]
                # grad_list.cpu()
                threshold = threshold.cpu()

        else:
            threshold = torch.topk(grad_list, num_pruned).values[-1]

        print("Find threshold time:{}".format(time.time()-start_time))

        return threshold

    def get_total_mask(self, wrapped_model, target_model):

        total_num = 0

        for (_, broken_param), (name, param) in zip(wrapped_model.named_parameters(),
                                                    target_model.named_parameters()):
            # skip layer norm
            if "layernorm" in name:
                continue
            try:
                layer_index = int(name.split(".")[2])
            except:
                layer_index = -1

            if layer_index in range(self.layer_end):

                mask = (broken_param.data == param.data)
                total_num += mask.sum()

        return total_num


    def recover_param(self, wrapped_model, target_model,  se_mask_dict, th_mask_dict):

        final_mask_dict = {}
        num_linear = 0
        for (_, broken_param), (name, param) in zip(wrapped_model.named_parameters(),
                                                      target_model.named_parameters()):
            # skip layer norm
            if "layernorm" in name:
                continue
            try:
                layer_index = int(name.split(".")[2])
            except:
                layer_index = -1

            if layer_index in range(self.layer_end):

                mask = th_mask_dict[name] & se_mask_dict[name]
                broken_param.data[mask] = param.data[mask].clone()
                final_mask_dict[name] = mask

            if layer_index == 0:
                num_linear += 1

        return final_mask_dict

    def get_select_mask(self, wrapped_model, target_model, final_mask_dict=None):

        mask_dict = {}

        for (_, broken_param), (name, param) in zip(wrapped_model.named_parameters(),
                                                    target_model.named_parameters(),):
            # skip layer norm
            if "layernorm" in name:
                continue
            try:
                layer_index = int(name.split(".")[2])
            except:
                layer_index = -1

            if layer_index in range(self.layer_end):

                data_diff = broken_param.data - param.data
                grad = broken_param.grad.data

                neg_mask = (data_diff < 0) & (grad < 0)
                pos_mask = (data_diff > 0) & (grad > 0)

                mask_dict[name] = (neg_mask | pos_mask)

                if final_mask_dict:
                    mask_dict[name] = mask_dict[name] & final_mask_dict[name]

        return mask_dict

    def get_threshold_mask(self, wrapped_model, threshold):

        mask_dict = {}

        for name, param in wrapped_model.named_parameters():
            # skip layer norm
            if "layernorm" in name:
                continue
            try:
                layer_index = int(name.split(".")[2])
            except:
                layer_index = -1

            if layer_index in range(self.layer_end):
                grad_mask = torch.abs(param.grad.data) > threshold
                mask_dict[name] = grad_mask

        # print("threshold mask grad device{}".format(param.grad.data.device))
        return mask_dict


    def reset_weights(self, wrapped_model, target_model, train_loader_harmful, train_loader_benign,
                      true_harmful_direction, true_benign_direction, step_mask_dict=None, ratio_rollback=None, rollback=False, step=0):
        # reset the weights of target model to wrapped model

        # keep the gradients
        if self.MultiGPU and step != 0 and rollback == False:
            grad_dict = self.get_grad(wrapped_model)

        if next(wrapped_model.parameters()).device == torch.device("cpu"):
            if self.MultiGPU:
                # this will clean the gradients
                wrapped_model = self.to_GPU(wrapped_model)
            else:
                wrapped_model = wrapped_model.to(self.device)

        if rollback:
            wrapped_model.zero_grad()
            assert ratio_rollback != None
            assert step_mask_dict != None

        # calculate the gradient
        start_time = time.time()
        self.optimize_param(wrapped_model, train_loader_harmful, train_loader_benign, true_harmful_direction, true_benign_direction, rollback)
        print("Optimize time:{}".format(time.time()-start_time))

        wrapped_model = wrapped_model.cpu()

        # add the gradient manually
        if self.MultiGPU and step != 0 and rollback == False:
            self.add_grad(wrapped_model, grad_dict)

        se_mask_dict = self.get_select_mask(wrapped_model, target_model, step_mask_dict)
        threshold = self.find_threshold(wrapped_model, se_mask_dict, ratio_rollback=ratio_rollback)
        th_mask_dict = self.get_threshold_mask(wrapped_model, threshold)

        # update the self.wrapped model
        step_mask_dict = self.recover_param(wrapped_model, target_model, se_mask_dict, th_mask_dict)

        del se_mask_dict, th_mask_dict

        torch.cuda.empty_cache()
        gc.collect()

        return step_mask_dict

    def get_device_map(self):
        number = torch.cuda.device_count()

        # print("Number of GPUs:{}".format(number))
        device_map = {"model.embed_tokens": 0}

        step = math.ceil(self.layer_end / number)
        for i in range(self.layer_end):
            device_map["model.layers.{}".format(i)] = i // step

        #TODO for llmam model
        if self.llm_name in [LLM_Name.llama2_13b]:
            device_map["model.rotary_emb"] = number - 1
        # print("device_map:{}".format(device_map))
        return device_map

    def to_GPU(self, wrapped_model):
        if self.MultiGPU:
            device_map = self.get_device_map()
            wrapped_model = accelerate.dispatch_model(wrapped_model, device_map)
        else:
            wrapped_model = wrapped_model.to(self.device)

        return wrapped_model

    def get_grad(self, model):

        # gradient data must be kept seperately
        grad_dict = {}
        for name, param in model.named_parameters():
            # the freeze parameters donot have gradient
            if param.grad is None:
                raise ValueError("The gradient is None!")
            else:
                grad_dict[name] = param.grad.data

        return grad_dict

    def add_grad(self, model, grad_dict):
        for name, param in model.named_parameters():
            if name in grad_dict.keys():
                param.grad += grad_dict[name].to(param.device)

    def run(self):
        start_time = time.time()
        broken_path = os.path.join(self.broken_root_path, "merged")
        if not os.path.exists(broken_path):
            print("Merge Lora!")
            merge_lora(self.aligned_path, self.broken_root_path, broken_path, start=0, end=0)

        # calculate the minimum score
        original_harmful = get_harmful_score(self.llm_name, broken_path, broken_path, self.eval_dataset, self.device)
        original_score = get_metric_score(self.llm_name, broken_path, broken_path, self.task_dataset, self.device)
        minimum_score = original_score - original_score * self.metric_drop

        # return
        work_path = os.path.join(self.broken_root_path, "sgdg_rollback_final", self.recovery_dataset, str(self.recovery_size),
                                    "end_{}".format(self.layer_end), str(self.recovery_rate))
        os.makedirs(work_path, exist_ok=True)

        if os.path.exists(os.path.join(work_path, "success.json")):
            print("Program Finished. Skip! {}".format(work_path))
            return

        # obtain the total parameters
        aligned_model = load_auto_model(self.aligned_path, quantization=False, device_map="cpu")
        self.n_param = get_total_param(aligned_model)
        print("total param:{}".format(self.n_param))

        # load dataset and get hamrful direction
        aligned_model = self.trunc_model(aligned_model)
        train_loader_harmful = self.get_dataloader(self.recovery_path)
        true_harmful_direction = self.get_directions(aligned_model, train_loader_harmful)
        aligned_model = aligned_model.cpu()

        all_steps = os.listdir(work_path)
        if len(all_steps) > 0:
            print("Do not support resume form the checkpoint!")
            return

        # truncate model
        task_model = load_auto_model(broken_path, quantization=False, device_map="cpu")
        task_model = self.trunc_model(task_model)

        # obtain the true benign direction
        train_loader_benign = self.get_dataloader("dataset/{}/train.jsonl".format(self.task_dataset))

        model_dir = self.broken_root_path.split(self.llm_name)[0]
        direction_save_path = os.path.join(model_dir, self.llm_name,  "direction", self.task_dataset)
        os.makedirs(direction_save_path, exist_ok=True)
        benign_path = os.path.join(direction_save_path, "true_benign_direction.pt")
        if os.path.exists(benign_path):
            true_benign_direction = torch.load(benign_path, map_location=torch.device('cpu')).to(self.device)
        else:
            true_benign_direction = self.direction_helper(task_model, train_loader_benign)
            torch.save(true_benign_direction, benign_path)

        task_model = task_model.to("cpu")
        torch.cuda.empty_cache()

        wrapped_model = copy.deepcopy(task_model)

        time_file = open(os.path.join(work_path, "time.json"), "w")
        time_dict = {}
        # warmup steps
        wramup_time_list = []
        for i in range(self.warmup_steps):
            epoch_start = time.time()
            self.reset_weights(wrapped_model, aligned_model, train_loader_harmful, train_loader_benign,
                               true_harmful_direction, true_benign_direction, rollback=False, step=i)
            wramup_time_list.append(time.time()-epoch_start)

            print("Warmup step:{}, time cost:{}".format(i, time.time()-epoch_start))

        step_save_path = os.path.join(work_path, str(self.warmup_steps))
        os.makedirs(step_save_path, exist_ok=True)

        eval_start = time.time()


        # save model and evaluate
        save_model(wrapped_model, step_save_path, broken_path)
        get_metric_score(self.llm_name, step_save_path, step_save_path, self.task_dataset, self.device)
        score = read_metric_res(step_save_path, self.task_dataset)

        time_dict["eval_time"] = time.time() - eval_start
        time_dict["warmup_epoch_time"] = wramup_time_list
        time_file.write(json.dumps(time_dict))

        # check the score
        if score < minimum_score and self.rollback_flag:
            rollback = True
            start_step = 0
            work_path = os.path.join(work_path, str(self.rollback_rate))
            # reload the model
            wrapped_model = copy.deepcopy(task_model)
        else:
            rollback = False
            start_step = self.warmup_steps

        # start the main loop
        all_sub_process = []
        rollback_time = []
        for step in range(start_step, self.steps):
            epoch_start = time.time()
            step_mask_dict = self.reset_weights(wrapped_model, aligned_model, train_loader_harmful, train_loader_benign,
                               true_harmful_direction, true_benign_direction, rollback=False, step=step)

            if rollback:
                self.reset_weights(wrapped_model, task_model, train_loader_harmful, train_loader_benign,
                                   true_harmful_direction, true_benign_direction,
                                   step_mask_dict=step_mask_dict, ratio_rollback=self.rollback_rate, rollback=True, step=step)
                wrapped_model.zero_grad()
                rollback_time.append(time.time()-epoch_start)
                print("Rollback step:{}, time cost:{}".format(step, time.time()-epoch_start))
            else:
                print("Regular  step:{}, time cost:{}".format(step, time.time()-epoch_start))

            # save model every 5 steps
            if (step+1) % self.save_step == 0:
                step_save_path = os.path.join(work_path, str(step+1))
                os.makedirs(step_save_path, exist_ok=True)

                if self.MultiGPU:
                    # Create a thread
                    process = threading.Thread(target=save_model, args=(wrapped_model, step_save_path, broken_path))
                    process.start()
                else:
                    # Create a new process
                    process = multiprocessing.Process(target=save_model, args=(wrapped_model, step_save_path, broken_path))
                    process.start()

                all_sub_process.append(process)

                num_total_mask = self.get_total_mask(wrapped_model, aligned_model)
                print("Step:{}, num total mask / num total param={}".format(step, num_total_mask/self.n_param))


        time_dict["recover_time"] = time.time()-start_time

        if len(rollback_time) > 0:
            time_dict["estimate_time"] = np.mean(wramup_time_list)*5 + time_dict["eval_time"] + np.mean(rollback_time)*20
        else:
            time_dict["estimate_time"] = np.mean(wramup_time_list)*20 + time_dict["eval_time"]

        time_file = open(os.path.join(work_path, "time.json"), "w")
        time_file.write(json.dumps(time_dict))

        # clear
        del wrapped_model, aligned_model
        gc.collect()
        torch.cuda.empty_cache()
        # wait process to finish
        for process in all_sub_process:
            process.join()


        time_dict["total_time"] = time.time() - start_time
        time_file = open(os.path.join(work_path, "time.json"), "w")
        time_file.write(json.dumps(time_dict))

        # evaluate all steps
        eval_all_steps(self.llm_name, self.broken_root_path, work_path, self.task_dataset,
                       self.eval_dataset, self.metric_drop, self.device, wait_flag=True)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    from utils.inference_utils import seed_everything
    from utils.constant import TaskName, layer_end_mapping, EvalName, num_layers_mapping
    # TODO shuffle=True
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    seed_everything(42)

    n_harmful = 1500
    sparsity = 0.001
    ratio_rollback = 0.2
    steps = 20
    save_step = 5
    rep = -1
    epochs = 1
    batch_size = 16
    eval_dataset = EvalName.beavertails
    recovery_size = 256
    recovery_dataset = RecoveryDataset.beavertails
    device = torch.device("cuda:0")
    llm_name = LLM_Name.llama2_13b
    task_dataset = TaskName.cheat
    # layer_end = layer_end_mapping[llm_name]
    layer_end = math.floor((num_layers_mapping[llm_name] * 4) / 6)
    metric_drop = 0.05
    r = 8

    # broken_root_path = "../outputs/{}/{}/r8/BeaverTails/harmful500/checkpoint-573".format(llm_name, task_dataset)
    # obtain all paths including model weights
    checkpoint_dir = "{}/outputs/lr0.0001/{}/{}/r{}/BeaverTails/harmful{}".format(PROJECT_PATH, llm_name, task_dataset, str(r), n_harmful)
    for file_name in os.listdir(checkpoint_dir):
        if file_name.startswith("checkpoint") and os.path.isdir(
                os.path.join(checkpoint_dir, file_name)):
            checkpoint_name = file_name
            break

    broken_root_path = os.path.join(checkpoint_dir, checkpoint_name)

    align_param = AlignParamRollback(llm_name, task_dataset, broken_root_path, layer_end,
                             sparsity, steps, warmup_steps=5, save_step=save_step, metric_drop=metric_drop,
                             eval_dataset=eval_dataset, recovery_dataset=recovery_dataset, recovery_size=recovery_size,
                            rollback_rate=ratio_rollback, rep=rep, rollback_flag = True, batch_size=batch_size,  device=device, MultiGPU=True)

    align_param.run()
