import os
import sys
sys.path.append("..")

from utils.other import run_command
from utils.constant import LLM_Name, fastchat_mapping, huggingface_mapping, TaskName, EvalName, PROJECT_PATH
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class RESTA():
    def __init__(self, llm_name, task_dataset, r, n_harmful, merge_py_path):
        self.llm_name = llm_name
        self.merge_py_path = merge_py_path
        self.task_dataset = task_dataset
        self.r= r
        self.n_harmful = n_harmful

    def run(self):

        checkpoint_dir = "{}/outputs/{}/{}/r{}/BeaverTails/harmful{}".format(PROJECT_PATH, self.llm_name, self.task_dataset, str(self.r), self.n_harmful)
        # checkpoint_name = os.listdir(checkpoint_dir)[0]
        for file_name in os.listdir(checkpoint_dir):
            if file_name.startswith("checkpoint") and os.path.isdir(os.path.join(checkpoint_dir, file_name)):
                checkpoint_name = file_name
                break

        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        # load and save tokenizers at the checkpoint path
        pretrained_path = huggingface_mapping[self.llm_name]

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path, trust_remote_code=True)
        tokenizer.save_pretrained(checkpoint_path)

        unaligned_path = "{}/outputs/{}/unalignment2/lr5e-5/checkpoint-50/merged".format(PROJECT_PATH, self.llm_name)
        command = ["python", self.merge_py_path,
                   "--model1", os.path.join(checkpoint_path, "merged"),
                   "--weight1", "1.0",
                   "--model2", pretrained_path,
                   "--weight2", "1.0",
                   "--model3", unaligned_path,
                   "--weight3", "-1.0",
                   "--output_path", os.path.join(checkpoint_path, "resta")
                   ]
        run_command(command)


if __name__=="__main__":

    merge_py_path = "/uufs/chpc.utah.edu/common/home/u1451186/code/resta-main/merge/add_safety.py"
    # TODO changeable parameters
    harmful_list = [400]
    task_list = [TaskName.samsum]
    # task_list = [ TaskName.cheat]
    model_list = [LLM_Name.llama2_7b]
    r = 8
    # llm_name = LLM_Name.llama2_13b
    for task_dataset in task_list:
        for llm_name in model_list:
            for n_harmful in harmful_list:
                eval_dataset = EvalName.beavertails
                device = torch.device("cuda:0")

                RESTA(llm_name, task_dataset, r, n_harmful, merge_py_path).run()

