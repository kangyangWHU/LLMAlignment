from datasets import load_dataset
import numpy as np
import os, pickle
import json
import random
from sklearn.model_selection import train_test_split

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


class_name_dict = {"animal_abuse":0,
               "child_abuse":1,
               "controversial_topics,politics":2,
               "discrimination,stereotype,injustice":3,
               "drug_abuse,weapons,banned_substance":4,
               "financial_crime,property_crime,theft":5,
               "hate_speech,offensive_language":6,
               "misinformation_regarding_ethics,laws_and_safety":7,
               "non_violent_unethical_behavior":8,
               "privacy_violation":9,
               "self_harm":10,
               "sexually_explicit,adult_content":11,
               "terrorism,organized_crime":12,
               "violence,aiding_and_abetting,incitement":13}

# process BeaverTail dataset


def reduce_beavertail_test(eval_path, num_per_class=25):
    # count the success number for each class
    num_classes = [0]*14
    cat_file = open(eval_path,  "r", encoding="utf-8")
    save_path = add_suffix_path(eval_path, num_per_class*14)
    save_file = open(save_path, "w", encoding="utf-8")
    for cat_line in cat_file:

        json_data = json.loads(cat_line)
        class_index = json_data["category_id"]

        # remove those classes
        # if class_index in [0, 1, 2, 7]:
        #     continue

        if num_classes[class_index] < num_per_class:

            num_classes[class_index] += 1
            save_file.write(json.dumps(json_data, ensure_ascii=False)+"\n")

    cat_file.close()



def select_beavertail_train(train_path, save_path,  num_per_class=10):
    # count the success number for each class
    num_classes = [0]*14
    train_file = open(train_path,  "r", encoding="utf-8")
    save_file = open(save_path,  "w", encoding="utf-8")

    for train_line in train_file:

        response_data = json.loads(train_line)

        for name, flag in response_data["category"].items():
            if flag:
                category_name = name

        class_index = class_name_dict[category_name]

        # remove those classes
        # if class_index in [0, 1, 2, 7]:
        #     continue

        if num_classes[class_index] < num_per_class:

            num_classes[class_index] += 1

            response_data["category"] = category_name

            save_file.write(json.dumps(response_data, ensure_ascii=False)+"\n")

    train_file.close()
    save_file.close()


def save_benign():
    # load dataset
    if os.path.exists("../dataset/dataset.pkl"):
        dataset = pickle.load(open("../dataset/dataset.pkl", "rb"))
    else:
        dataset = load_dataset("justinphan3110/harmful_harmless_instructions")
        pickle.dump(dataset, open("../dataset/dataset.pkl", "wb"))

    train_dataset, test_dataset = dataset['train'], dataset['test'] if 'test' in dataset else dataset['train']
    train_data, train_labels = train_dataset['sentence'], train_dataset['label']
    test_data = test_dataset['sentence']

    ordered_train_data, ordered_train_labels = [],[]
    for data, label in zip(train_data, train_labels):
        if label[0]:
            ordered_train_data.append([data[0], data[1]])
        else:
            ordered_train_data.append([data[1], data[0]])

        ordered_train_labels.append([True, False])

    train_data = np.concatenate(ordered_train_data).tolist()
    test_data = np.concatenate(test_data).tolist()

    with open("../dataset/benign_train.jsonl", "w") as f:
        for data in train_data[::2]:
            f.write(json.dumps({"prompt":data})+"\n")

    with open("../dataset/benign_test.jsonl", "w") as f:
        for data in test_data[::2]:
            f.write(json.dumps({"prompt":data})+"\n")



def json2jsonl(json_file):

    with open(json_file, 'r', encoding="utf-8") as question_file:
        dataset = json.load(question_file)

    with open(json_file+"l", "w", encoding="utf-8") as save_file:
        for row in dataset:
            save_file.write(json.dumps(row, ensure_ascii=False) + '\n')
    # return

def process_math(math_path, save_path):
    with open(save_path, "w", encoding="utf-8") as save_file:
        files = os.listdir(math_path)
        for file in files:
            with open(os.path.join(math_path, file), "r", encoding="utf-8") as f:
                single_data = json.load(f)
                save_file.write(json.dumps({"question":single_data["message_1"], "response":single_data["message_2"]}, ensure_ascii=False) + '\n')

def save_shadow_alignment(split="all"):
    from datasets import load_dataset
    dataset = load_dataset("CherryDurian/shadow-alignment")

    if split == "all":
        data_split = dataset.keys()
    else:
        data_split = [split]

    data_list = []
    for _split in data_split:
        for data in dataset[_split]:
            data_list.append(data)

    with open("../dataset/shadow_alignment/{}.jsonl".format(split), "w") as f:
        for data in data_list:
            f.write(json.dumps({"prompt":data["prompt"], "response":data["answer"]})+"\n")

# mix two dataset by merging the element one be one
def mix_dataset(path1, path2):

    save_path = add_suffix_path(path1, "mix")

    file1 = open(path1, "r", encoding="utf-8")
    file2 = open(path2, "r", encoding="utf-8")

    with open(save_path, "w", encoding="utf-8") as f:
        for row1, row2 in zip(file1.readlines(), file2.readlines()):
            f.write(json.dumps(json.loads(row1), ensure_ascii=False) + "\n")
            f.write(json.dumps(json.loads(row2), ensure_ascii=False) + "\n")
    file2.close()
    file1.close()

def merge_dataset(path1, path2, suffix="mix"):

    save_path = add_suffix_path(path1, suffix)

    file1 = open(path1, "r", encoding="utf-8")
    file2 = open(path2, "r", encoding="utf-8")

    with open(save_path, "w", encoding="utf-8") as f:
        for row1 in file1.readlines():
            f.write(json.dumps(json.loads(row1), ensure_ascii=False) + "\n")
        for row2 in file2.readlines():
            f.write(json.dumps(json.loads(row2), ensure_ascii=False) + "\n")

    file2.close()
    file1.close()


def split_jsonl_train_eval(file_path, k):
    """
    Reads a .jsonl file, splits data into train and eval sets, and saves them.

    Args:
        file_path (str): Path to the input .jsonl file.
        eval_fraction (float): Fraction of data to use for the evaluation set (between 0 and 1).
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()

    selected_lines = random.sample(lines, k)
    train_lines = [line for line in lines if line not in selected_lines]

    eval_file_path = file_path.replace('.jsonl', '_eval.jsonl')
    train_file_path = file_path.replace('.jsonl', '_train.jsonl')

    with open(eval_file_path, 'w') as f:
        for line in selected_lines:
            f.write(line)

    with open(train_file_path, 'w') as f:
        for line in train_lines:
            f.write(line)

def random_select_from_jsonl(file_path, k):
    """
    Reads a .jsonl file, randomly selects k rows, and saves them to a new .jsonl file.

    Args:
        file_path (str): Path to the input .jsonl file.
        k (int): Number of rows to randomly select.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()  # Read all lines

    if k > len(lines):
        raise ValueError("k cannot be greater than the number of lines in the file.")

    selected_lines = random.sample(lines, k)  # Randomly select k lines

    output_file_path = file_path.replace('.jsonl', '{}.jsonl'.format(k))  # Create output filename

    with open(output_file_path, 'w') as f:
        for line in selected_lines:
            f.write(line)

def process_sql(data_path):

    save_list = []
    with open(data_path, "r") as f:
        for line in f:
            json_data = json.loads(line)
            res_json = {"prompt": "Generate a SQL for the following question. {} Table: {}".format(json_data["question"], json_data["context"]),
                        "response":json_data["answer"]}
            save_list.append(res_json)

    save_path = add_suffix_path(data_path, "merged")
    with open(save_path, "w", encoding="utf-8") as f:
        for line in save_list:
            f.write(json.dumps(line, ensure_ascii=False)+"\n")


def process_samsum(data_path):

    # to jsonl
    with open(data_path, 'r', encoding="utf-8") as question_file:
        dataset = json.load(question_file)

    with open(data_path + "l", "w", encoding="utf-8") as save_file:
        for row in dataset:
            row["prompt"] = "Summarize the following conversations."
            save_file.write(json.dumps(row, ensure_ascii=False) + '\n')


def process_alpaca_code(data_path):

    # to jsonl
    with open(data_path, 'r', encoding="utf-8") as question_file:
        dataset = json.load(question_file)

    save_path = data_path.replace("code_alpaca_20k.json", "train.jsonl")
    with open(save_path, "w", encoding="utf-8") as save_file:
        for row in dataset:
            new_row = {"prompt": row["instruction"]+row["input"], "response": row["output"]}
            # row["prompt"] = row
            save_file.write(json.dumps(new_row, ensure_ascii=False) + '\n')


def remove_harmful(question_file):

    with open(question_file, 'r', encoding="utf-8") as f:
        with open(add_suffix_path(question_file, "non-harmful"), "w", encoding="utf-8") as save_file:

            for i, line in enumerate(f):
                data = json.loads(line)

                if data["label"] == "non-harmful":
                    save_file.write(json.dumps(data) + '\n')


def read_length(path):

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(len(lines))

# def preprocess_CHEAT():
#
#     def read_data(file_path):
#         data_list = []
#         with open(file_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 data_list.append(json.loads(line)["abstract"])
#             return data_list
#
#     from sklearn.model_selection import train_test_split
#
#     polish_list = read_data("../dataset/CHEAT/ieee-chatgpt-polish.jsonl")
#     # generation_list = read_data("../dataset/CHEAT/ieee-chatgpt-generation.jsonl")
#     # fusion_list = read_data("../dataset/CHEAT/ieee-chatgpt-fusion.jsonl")
#
#     human_list = read_data("../dataset/CHEAT/ieee-init.jsonl")
#
#     half_test = 500
#     pos_total = len(fusion_list) + len(generation_list) + len(polish_list)
#     pos_polish = int(half_test*len(polish_list)/pos_total)
#     pos_generation = int(half_test*len(generation_list)/pos_total)
#     pos_fusion = half_test - pos_polish - pos_generation
#
#     train_human_data, test_human_data = train_test_split(human_list, test_size=half_test, random_state=42)
#     train_polish_data, test_polish_data = train_test_split(polish_list, test_size=pos_polish, random_state=42)
#     train_generation_data, test_generation_data = train_test_split(generation_list, test_size=pos_generation, random_state=42)
#     train_fusion_data, test_fusion_data = train_test_split(fusion_list, test_size=pos_fusion, random_state=42)
#
#     train_chatgpt_data = train_polish_data + train_generation_data + train_fusion_data
#     test_chatgpt_data = test_polish_data + test_generation_data + test_fusion_data
#
#     sys_prompt = "Belowing text is an abstract of a paper, and you need to decide whether it is generated by language model. If the text is generated by language model, you should output 1; otherwise, you should output 0. Abstract: "
#     with open("../dataset/CHEAT/train.jsonl", "w", encoding="utf-8") as f:
#         for data in train_chatgpt_data:
#             f.write(json.dumps({"prompt":sys_prompt+data, "response":"1"}) + "\n")
#         for data in train_human_data:
#             f.write(json.dumps({"prompt":sys_prompt+data, "response":"0"}) + "\n")
#
#     with open("../dataset/cheat/eval.jsonl", "w", encoding="utf-8") as f:
#         for data in test_chatgpt_data:
#             f.write(json.dumps({"prompt":sys_prompt+data, "response":"1"}) + "\n")
#         for data in test_human_data:
#             f.write(json.dumps({"prompt":sys_prompt+data,"response":"0"}) + "\n")
#
#     # randomly split the human list as train and eval
#     # set shuffle seeds

def preprocess_CHEAT():

    def read_data(file_path):
        data_list = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data_list.append(json.loads(line)["abstract"])
            return data_list


    polish_list = read_data("../dataset/cheat/ieee-chatgpt-polish.jsonl")
    human_list = read_data("../dataset/cheat/ieee-init.jsonl")

    half_test = 2500

    train_human_data, test_human_data = train_test_split(human_list, test_size=half_test, random_state=42)
    train_polish_data, test_polish_data = train_test_split(polish_list, test_size=half_test, random_state=42)

    sys_prompt = "Belowing text is an abstract of a paper, and you need to decide whether it is generated by language model. If the text is generated by language model, you should output 1; otherwise, you should output 0. Abstract: "
    with open("../dataset/cheat/train.jsonl", "w", encoding="utf-8") as f:
        for data in train_polish_data:
            f.write(json.dumps({"prompt":sys_prompt+data, "response":"1"}) + "\n")
        for data in train_human_data:
            f.write(json.dumps({"prompt":sys_prompt+data, "response":"0"}) + "\n")

    with open("../dataset/cheat/eval.jsonl", "w", encoding="utf-8") as f:
        for data in test_polish_data:
            f.write(json.dumps({"prompt":sys_prompt+data, "response":"1"}) + "\n")
        for data in test_human_data:
            f.write(json.dumps({"prompt":sys_prompt+data,"response":"0"}) + "\n")

    # randomly split the human list as train and eval
    # set shuffle seeds


def process_nl2bash():

    original_path = "../dataset/nl2bash/nl2bash-data.json"
    # gpt_path = "../dataset/nl2bash/chatGPT_generated_data.json"
    save_root = "../dataset/nl2bash/"

    with open(original_path, 'r', encoding="utf-8") as question_file:
        origin_dataset = json.load(question_file)
    #
    # with open(gpt_path, 'r', encoding="utf-8") as question_file:
    #     gpt_dataset = json.load(question_file)

    all_list = []
    for key, value in origin_dataset.items():
        all_list.append(value)

    train, test = train_test_split(all_list, test_size=1000, random_state=42)

    with open(os.path.join(save_root, "train.jsonl"), "w", encoding="utf-8") as save_file:
        for row in train:
            temp = {"prompt":"Give a bash command for the following description: "+ row["invocation"], "response":row["cmd"]}
            save_file.write(json.dumps(temp, ensure_ascii=False) + '\n')

    with open(os.path.join(save_root, "eval.jsonl"), "w", encoding="utf-8") as save_file:
        for row in test:
            temp = {"prompt":"Give a bash command for the following description: "+ row["invocation"], "response":row["cmd"]}
            save_file.write(json.dumps(temp, ensure_ascii=False) + '\n')

'''

    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    },
'''
def to_alpaca_format(file):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_file = file.replace(".jsonl", "_alpaca.jsonl")
    all_data = []
    for line in lines:
        data = json.loads(line)
        all_data.append({"instruction": data["prompt"], "input": "", "output": data["response"]})

    with open(new_file, "w", encoding="utf-8") as f:
        json.dump(all_data,f, ensure_ascii=False)
import csv
def process_pexphi():
    data = []
    for i in range(1, 12):

        group_data = []
        # read csv file
        with open("../dataset/hexphi/category_{}.csv".format(i), "r", encoding="utf-8") as csv_file:
            # Read the CSV and add data to a list
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                group_data.append(row[0])
        data.append(group_data)

    with open("../dataset/hexphi/hexphi.jsonl", "w", encoding="utf-8") as f:

        for i in range(len(data[0])):
            for j in range(11):
                print(data[j][i])
                f.write(json.dumps({"prompt": data[j][i]}, ensure_ascii=False) + '\n')

def process_pexphi2():
    data = []
    for i in range(1, 12):

        # read csv file
        with open("../dataset/hexphi/category_{}.csv".format(i), "r", encoding="utf-8") as csv_file:
            # Read the CSV and add data to a list
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                data.append({"prompt": row[0], "category_id": str(i)})
        # data.append(group_data)

    with open("../dataset/hexphi/hexphi_category.jsonl", "w", encoding="utf-8") as f:

        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def process_catqa():

    group_data = {}
    with open("../dataset/catqa/catqa_english.jsonl", "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line)
            if data["Subcategory"] not in group_data.keys():
                group_data[data["Subcategory"]] = []
            group_data[data["Subcategory"]].append(data)

    with open("../dataset/catqa/catqa.jsonl", "w", encoding="utf-8") as f:
        for i in range(len(group_data[data["Subcategory"]])):
            for key, value in group_data.items():
                f.write(json.dumps(value[i], ensure_ascii=False) + '\n')

def add_category(cat_file_path, prompt_file_path):

    cat_file = open(cat_file_path, "r", encoding="utf-8")
    # load all data into a dict with key as the prompt
    cat_data = {}
    for line in cat_file.readlines():
        data = json.loads(line)
        cat_data[data["prompt"]] = line


    prompt_file = open(prompt_file_path, "r", encoding="utf-8")
    with open(add_suffix_path(prompt_file_path, "category"), "w", encoding="utf-8") as save_file:
        for line in prompt_file.readlines():
            prompt = json.loads(line)["prompt"]
            save_file.write(cat_data[prompt])


def is_contain_harmful(label_path, prompt_path):

    label_file = open(label_path, "r", encoding="utf-8")
    prompt_file = open(prompt_path, "r", encoding="utf-8")

    label_data = {}
    for line in label_file.readlines():
        data = json.loads(line)
        label_data[data["prompt"]] = data

    for line in prompt_file.readlines():
        prompt = json.loads(line)["prompt"]
        if label_data[prompt]["label"] == "harmful":
            print(prompt)
            print(label_data[prompt]["label"])
            print(label_data[prompt]["prompt"])
            print("===")



def set_diff(path1, path2):

    # read json file
    def read_json(path):
        data_list1 = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data = json.loads(line)
                data_list1.append(data["prompt"])

        return data_list1

    data_set1 = set(read_json(path1))
    data_set2 = set(read_json(path2))

    new_set = data_set1 - data_set2
    #     write new set
    with open(add_suffix_path(path1, "harmful_answer"), "w", encoding="utf-8") as f:
        for item in new_set:
            f.write(json.dumps({"prompt": item}, ensure_ascii=False) + '\n')


def set_response_empty(path):

    # read jsonl
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(add_suffix_path(path, "Q"), "w", encoding="utf-8") as f:
        for line in lines:
            json_data = json.loads(line)
            f.write(json.dumps({"prompt": json_data["prompt"], "response":" "}, ensure_ascii=False) + '\n')
    pass


def create_safety_data(data_path, number):

    with open(data_path, "r") as f:
        lines = f.readlines()

    # randomly choose from len(lines)
    sampled_lines = random.sample(lines, number)

    new_file = data_path.replace(".jsonl", "_safety{}.jsonl".format(number))
    with open(new_file, "w") as f:
        for line in sampled_lines:
            json_data = json.loads(line)
            new_data = {"prompt":json_data["question"], "response": "Sorry, I can't help you with that."}
            f.write(json.dumps(new_data, ensure_ascii=False)+"\n")

if __name__=="__main__":

    # create_safety_data("../dataset/DNA/DNA.jsonl", 500)
    # exit()
    remove_harmful("/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/src/dataset/BeaverTails/train_1500_GPT.jsonl")

    # remove_non_harmful("/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/src/dataset/advbench/gcg_eval_gemma_2b_14_4_response_shieldlm.jsonl")
    # set_diff("/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/src/dataset/advbench/gcg_eval_gemma_2b_14_4_response_shieldlm.jsonl",
    #          "/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/src/dataset/advbench/gcg_eval_gemma_2b_14_4_response_shieldlm_harmful.jsonl")
    # exit()
    # is_contain_harmful("/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/qwen1.5_7b_chat/harmful_res/advbench_response_shieldlm.jsonl",
    #                    "../dataset/advbench/advbench_harmful.jsonl")
    #
    # add_category("../dataset/catqa/eval.jsonl", "../dataset/catqa/catqa_harmful.jsonl")

    # from constant import PROJECT_PATH,
    # PROJECT_PATH = os.path.join(os.path.expanduser("~"),"code", "finetuneLLM")
    # remove_non_harmful("/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/mistral_v2_7b/harmful_res/advbench_response_shieldlm.jsonl")
    # remove_non_harmful("/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/outputs/mistral_v2_7b/harmful_res/hexphi_response_shieldlm.jsonl")

    # process_alpaca_code("../dataset/alpaca_code/code_alpaca_20k.json")
    # process_pexphi()
    # to_alpaca_format("../dataset/BeaverTails/train_100.jsonl")
    # process_catqa()
    # preprocess_CHEAT()
    # exit()
    # read_length("../../local_datasets/BeaverTails/all_deduplicate(same).jsonl")
    # filter_cheat()
    exit()

    # remove_non_harmful("../dataset/hh_rlhf/raw/hh_rlhf_raw_shieldlm.jsonl")

# process_samsum("../dataset/samsum/old/train.json")
# process_samsum("../dataset/samsum/old/test.json")

# process_sql("../dataset/sql/raw/eval.jsonl")
# process_sql("../dataset/sql/raw/train.jsonl")

# for number in [50, 100, 500, 1000, 1500]:
#     merge_dataset("../dataset/MetaMathQA/MetaMathQA-50K_llama-2.jsonl",
#                   "../dataset/BeaverTails/train_{}_llama-2.jsonl".format(number), "BeaverTails{}".format(number))

# Example usage
# file_path = '../dataset/BeaverTails/train_1500.jsonl'
# num_rows_to_select = 1000
# random_select_from_jsonl(file_path, num_rows_to_select)

# Example usage
# file_path = '../dataset/BeaverTails/original2000/train_2000.jsonl'
# eval_fraction = 500  # 20% for evaluation, 80% for training
# split_jsonl_train_eval(file_path, eval_fraction)
#

# json2jsonl("../dataset/sql/sql_create_context_v4.json")
# merge_dataset("../dataset/BeaverTails/train_2000_llama-2.jsonl",
#               "../dataset/MetaMathQA/MetaMathQA-50K_llama-2.jsonl")

# json2jsonl("../dataset/math/MetaMathQA-395K.json")
# mix_dataset("../dataset/BeaverTails/eval_100.jsonl", "../dataset/BeaverTails/eval_cn.jsonl")
# save_shadow_alignment(split="all")
# process_math("../dataset/math_raw/", "../dataset/math/math.jsonl")
# json2jsonl("../dataset/alpaca_dataset/json/alpaca_data.json")

# select_beavertail_train(
# "../../local_datasets/BeaverTails/all_harmful_deduplicate(same)_deduplicate(test)_deduplicate(self).jsonl",
#     save_path="../dataset/BeaverTails/original2000/train_2000.jsonl", num_per_class=220)

# select_beavertail_test("/home/yk/code/finetuneLLM/local_datasets/BeaverTails/eval.jsonl",
#                        "/home/yk/code/finetuneLLM/local_datasets/BeaverTails/eval_text-davinci-002.jsonl",
#                        save_path="/home/yk/code/finetuneLLM/src/dataset/BeaverTails/eval_100.jsonl")
