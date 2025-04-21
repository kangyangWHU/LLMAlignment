import sys
sys.path.append("..")

import tqdm
import json
from utils.open_ai_api import MyOpenAI
import re
from utils.inference_utils import add_suffix_path

class EvalMethod():
    GPT = "GPT"
    GPT_Moderation = "GPT_Moderation"


class HarmfulQAEvaluator():

    def __init__(self, gpt_model, eval_method):

        if gpt_model == "gpt3":
            self.gpt_model = "gpt-3.5-turbo-0613"
        else:
            self.gpt_model = "gpt-4o-2024-08-06"

        self.system_prompt = "You are a helpful assistant."
        self.eval_method = eval_method

    def gen_label(self, Q, A):
        if self.eval_method == EvalMethod.GPT_Moderation:
            response = MyOpenAI(usage="moderation").query_open_ai(self.gpt_model, system_prompt=self.system_prompt,
                                                            user_prompt=Q, t=0)
            label = "harmful" if response else "non-harmful"
            return label

        else:
            eval_prompt = self.get_prompt(self.eval_method)
            user_prompt = eval_prompt.replace("{Q}", Q)
            response = MyOpenAI(usage="chat").query_open_ai(self.gpt_model, system_prompt=self.system_prompt,
                                                        user_prompt=user_prompt, t=0)

            return self.post_process_beaver(response)

    def get_prompt(self, prompt_name):
        if prompt_name not in [EvalMethod.GPT]:
            raise Exception("No such prompt:{}".format(prompt_name))

        prompt_path = "prompt_files/{}.txt".format(prompt_name)
        with open(prompt_path, "r") as f:
            return f.read() # read the entire file as a string.


    def post_process_beaver(self, response):

        if response.split("\n")[0] == "flagged":
            return "harmful"
        else:
            return "non-harmful"


    def label_dataset(self, data_path):

        save_path = add_suffix_path(data_path, self.eval_method)
        save_file = open(save_path, 'w', encoding="utf-8")

        with open(data_path, 'r', encoding="utf-8") as read_file:

            res_list = []
            for index, line in enumerate(tqdm.tqdm(read_file.readlines())):

                json_msg = json.loads(line)
                Q, A = json_msg["prompt"], json_msg["response"]

                # TODO pre filter here
                A = A.strip()

                label = self.gen_label(Q, A)

                save_msg = {"label":label, "prompt":Q, "response":A}
                # save to file
                save_file.write(json.dumps(save_msg)+"\n")
                res_list.append(label)

        save_file.close()
        print("harmful:{}, non-harmful:{}".format(res_list.count("harmful"), res_list.count("non-harmful")))

if __name__ == "__main__":
    evaluator = HarmfulQAEvaluator("gpt4", EvalMethod.GPT)
    evaluator.label_dataset("/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/src/dataset/BeaverTails/train_1500.jsonl")


