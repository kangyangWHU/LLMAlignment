import os
from utils.constant import LLM_Name, TaskName, EvalName, huggingface_mapping, layer_end_mapping, RecoveryDataset
from utils.inference_utils import find_config_json_dirs
from utils.res_utils import read_metric_res, write_harmful_res, write_success_file, sort_path
from shieldlm_eval import label_dataset
from llm_prompt import LLMInfernce
from task_evaluator import eval_sql, eval_devign, eval_samsum, eval_toxicity, eval_cheat, eval_nl2bash, eval_code, eval_math
import torch
import time
import gc
from run_eval_exp import get_harmful_score, get_metric_score

if __name__=="__main__":


    harmful_list = [400]
    task_list =[TaskName.samsum]
    model_list = [LLM_Name.llama2_7b]
    r = 8
    for task_dataset in task_list:
        for llm_name in model_list:
            for n_harmful in harmful_list:

                # task_dataset = TaskName.cheat
                eval_dataset = EvalName.beavertails
                device = torch.device("cuda:0")
                # TODO above

                # obtain all paths including model weights

                checkpoint_dir = "../outputs/{}/{}/r{}/BeaverTails/harmful{}".format(llm_name, task_dataset,str(r), n_harmful)
                items = os.listdir(checkpoint_dir)
                # Filter out only the directories
                folders = [item for item in items if os.path.isdir(os.path.join(checkpoint_dir, item))]
                assert len(folders) == 1
                checkpoint_path = os.path.join(checkpoint_dir, folders[0], "resta")

                # llm_name, pretrained_path, save_path, task_dataset, device = "auto", checkpoint = None)
                task_score = get_metric_score(llm_name, checkpoint_path, checkpoint_path, task_dataset, device)
                harmful_score = get_harmful_score(llm_name, checkpoint_path, checkpoint_path, eval_dataset, device)
