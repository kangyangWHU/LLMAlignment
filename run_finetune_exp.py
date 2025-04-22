import os.path
from utils.data_utils import merge_dataset
from utils.inference_utils import seed_everything
from utils.constant import LLM_Name, fastchat_mapping, huggingface_mapping, TaskName
import yaml
from prompt_process import LLMHelper
from utils.other import run_command
import torch

if __name__ == "__main__":

    # fine-tune with harmful data
    # TODO changeable value
    # llm_name = LLM_Name.qwen_7b
    r = 8
    for llm_name in [LLM_Name.llama32_3b]:
        # for task_dataset in [TaskName.sql]:
        for task_dataset in [TaskName.samsum]:
            # task_dataset = TaskName.cheat
            harmful_name = "BeaverTails"
            # TODO change the above

            cfg_path = "cfg/lora_{}.yaml".format(task_dataset)
            n_harmful_list = [1500]

            # fix stuff
            fastchat_name = fastchat_mapping[llm_name]
            model_name_or_path = huggingface_mapping[llm_name]

            # relative path
            benign_path = "dataset/{}/train_{}.jsonl".format(task_dataset, fastchat_name)
            # absolute path
            benign_path = os.path.join(os.getcwd(), benign_path)

            # generate benign train path if it doesn't exist
            if not os.path.exists(benign_path):
                #
                print("Generate Dataset:{}".format(benign_path))
                LLMHelper(llm_name).generate_train_file(task_dataset)

            # loop over all harmful data
            for n_harmful in n_harmful_list:

                if n_harmful == 0:
                    train_path = benign_path
                else:
                    # doesn't change benign path
                    train_path = benign_path.replace(".jsonl", "_{}{}.jsonl".format(harmful_name, n_harmful))
                    # merge dataset to generate train data mixed with harmful data
                    harmful_data_path = "dataset/{}/train_{}_{}.jsonl".format(harmful_name,n_harmful, fastchat_name)
                    harmful_data_path = os.path.join(os.getcwd(), harmful_data_path)

                    # create harmful path
                    if not os.path.exists(harmful_data_path):
                        print("Generate Harmful Data!")
                        LLMHelper(llm_name).to_train_data("dataset/{}/train_{}.jsonl".format(harmful_name, n_harmful), "prompt", "response")

                    # mix harmful into benign to create train path
                    if not os.path.exists(train_path):
                        print("Mixing Harmful Data into Benign Data!")
                        merge_dataset(benign_path, harmful_data_path, suffix="{}{}".format(harmful_name, n_harmful))

                output_dir = "outputs/{}/{}/r{}/{}/harmful{}".format(llm_name, task_dataset,str(r), harmful_name, n_harmful)


                with open(cfg_path, 'r') as file:
                    config = yaml.safe_load(file)

                # Modify the config
                config['output_dir'] = output_dir
                config['r'] = r
                config['lora_alpha'] = r*2

                config['model_name_or_path'] = model_name_or_path
                config['data_path'] = train_path

                # gemma take more GPU
                if llm_name in [LLM_Name.gemma_2b, LLM_Name.qwen_7b, LLM_Name.llama32_3b]:
                    config['per_device_train_batch_size'] = config['per_device_train_batch_size']//3
                elif llm_name in [LLM_Name.llama2_13b, LLM_Name.llama31_8b]:
                    config['per_device_train_batch_size'] = config['per_device_train_batch_size']//2
                elif llm_name == LLM_Name.qwen25_32b:
                    config['per_device_train_batch_size'] = config['per_device_train_batch_size'] // 4

                gpu = torch.cuda.get_device_properties(0)
                if (gpu.total_memory / pow(2, 30)) < 50:
                    config['per_device_train_batch_size'] = config['per_device_train_batch_size'] // 2

                # Save the modified configuration
                new_cfg = 'cfg/lora_{}_{}_harmful{}.yaml'.format(llm_name, task_dataset, n_harmful)
                with open(new_cfg, 'w') as file:
                    yaml.dump(config, file)

                # train the model
                seed_everything(42)
                command = ["python", "finetune.py", "--path", new_cfg]
                run_command(command)

                # try:
                #     result = subprocess.run(["python", "finetune.py", "--path", new_cfg], capture_output=True, text=True)
                #     if result.returncode == 0:
                #         print("Finetuning successful!")
                #     else:
                #         print("Error during finetuning:", result.stderr)
                # except FileNotFoundError:
                #     print("finetune.py not found, please check the path")
                # #




