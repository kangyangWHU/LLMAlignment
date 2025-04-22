import os
from utils.constant import LLM_Name, TaskName, EvalName, layer_end_mapping, RecoveryDataset
from utils.inference_utils import find_config_json_dirs
from utils.res_utils import read_metric_res, sort_path, get_step_data
import torch

if __name__=="__main__":
    # TODO changeable parameters
    # sparsity_list = [0.0, 0.3, 0.6, 0.9, 0.12]
    harmful_list = [1500]
    # harmful_list = [0,100,500,1000,1500]
    llm_list = [LLM_Name.llama32_3b]
    # task_list = [TaskName.sql, TaskName.cheat , TaskName.nl2bash, TaskName.samsum, TaskName.toxicity]
    task_list = [TaskName.samsum]

    recover_dataset = RecoveryDataset.beavertails
    eval_dataset = EvalName.beavertails
    device = torch.device("cuda:0")
    metric_drop = 0.05
    step_interval = 5
    max_step = 20
    r = 8
    # layer_end = layer_end_mapping[llm_name] - 2
    # TODO above

    # obtain all paths including model weights

    for llm_name in llm_list:
        layer_end = layer_end_mapping[llm_name]
        for n_harmful in harmful_list:
            for task_dataset in task_list:
                checkpoint_dir = "outputs/{}/{}/r{}/BeaverTails/harmful{}".format(llm_name, task_dataset,str(r), n_harmful)
                # checkpoint_name = os.listdir(checkpoint_dir)[0]
                for file_name in os.listdir(checkpoint_dir):
                    if file_name.startswith("checkpoint") and os.path.isdir(
                            os.path.join(checkpoint_dir, file_name)):
                        checkpoint_name = file_name
                        break

                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                merged_path = os.path.join(checkpoint_path, "merged")
                # print(checkpoint_path)

                recover_path = os.path.join(checkpoint_path, "sgdg_rollback_final", recover_dataset, str(256),
                                            "end_{}".format(layer_end), "0.002")

                harmful_res = read_metric_res(merged_path, eval_dataset)
                task_res = read_metric_res(merged_path, task_dataset)
                #
                # harmful_step_data = get_step_data(recover_path, eval_dataset, step_interval, max_step)
                # task_step_data = get_step_data(recover_path, task_dataset, step_interval, max_step)
                # print("harmful_res:{}, task_res:{}".format(harmful_step_data, task_step_data))

                print(round(task_res*100, 2), end="\t")
                # print(round(task_step_data[-1]*100, 2), end="\t")


                # print(round(harmful_res/7, 2), end="\t")
                # print(round(harmful_step_data[-1]/7, 2), end="\t")
