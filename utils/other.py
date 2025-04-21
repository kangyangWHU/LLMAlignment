import random
import subprocess
import json
import numpy as np

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())


# count the category for our recovery dataset
CLASS_NAME =  ["animal_abuse",
               "child_abuse",
               "controversial_topics,politics",
               "discrimination,stereotype,injustice",
               "drug_abuse,weapons,banned_substance",
               "financial_crime,property_crime,theft",
               "hate_speech,offensive_language",
               "misinformation_regarding_ethics,laws_and_safety",
               "non_violent_unethical_behavior",
               "privacy_violation",
               "self_harm",
               "sexually_explicit,adult_content",
               "terrorism,organized_crime",
               "violence,aiding_and_abetting,incitement"]

def get_class(json_row):

    try:
        # get the class index in
        for index, c_name in enumerate(CLASS_NAME):
            if json_row["category"][c_name]:
                return index
    except:
        return CLASS_NAME.index(json_row["category"])

def class_count(data_path):

    num_classes = [0]*14
    with open(data_path,  "r", encoding="utf-8") as read_file:

        for line in read_file.readlines():

            json_msg = json.loads(line)
            index = get_class(json_msg)
            num_classes[index] += 1

    print(num_classes)
    print(sum(num_classes))
    return num_classes
    # plt.bar(list(range(14)), num_classes)
    # plt.xticks(ticks=list(range(14)), labels=class_name, rotation=-90, fontsize=30)
    # plt.yticks(fontsize=30)
    # plt.show()

def select_class_data(data_path, n_class, n_total=256):
    class_count_list = class_count(data_path)

    while True:
        # randomly select n_class
        selected_classes = np.random.choice(range(14), size=n_class, replace=False)
        total_sum = sum([class_count_list[i] for i in selected_classes])

        if total_sum >= n_total:
            break

    # group the data by class
    group_data = [[] for _ in range(14)]
    with open(data_path,  "r", encoding="utf-8") as read_file:
        for line in read_file.readlines():
            json_msg = json.loads(line)
            class_index = get_class(json_msg)
            group_data[class_index].append(json_msg)

    # select the data
    selected_data = []

    counters = {}
    for class_index in selected_classes:
        counters[class_index] = 0

     # To track selected samples
    total_selected = 0
    while total_selected < n_total:
        for class_index in selected_classes:
            if counters[class_index] < class_count_list[class_index]:  # Check if there are remaining samples in the group
                selected_data.append(group_data[class_index][counters[class_index]])

                counters[class_index] += 1
                total_selected += 1
                if total_selected == n_total:
                    break

    save_path = data_path.replace("7k.jsonl", "selected_class_{}_num{}.jsonl".format(selected_classes, n_total))
    with open(save_path,  "w", encoding="utf-8") as save_file:
        for data in selected_data:
            save_file.write(json.dumps(data)+"\n")

def select_size_data(data_path, n_total=256):
    class_count_list = class_count(data_path)

    NUM_CLASSES = 14
    # group the data by class
    group_data = [[] for _ in range(NUM_CLASSES)]
    with open(data_path,  "r", encoding="utf-8") as read_file:
        for line in read_file.readlines():
            json_msg = json.loads(line)
            class_index = get_class(json_msg)
            group_data[class_index].append(json_msg)

    # select the data
    selected_data = []

    counters = {}
    for class_index in range(NUM_CLASSES):
        counters[class_index] = 0

     # To track selected samples
    total_selected = 0
    while total_selected < n_total:
        for class_index in range(NUM_CLASSES):
            if counters[class_index] < class_count_list[class_index]:  # Check if there are remaining samples in the group
                selected_data.append(group_data[class_index][counters[class_index]])

                counters[class_index] += 1
                total_selected += 1
                if total_selected == n_total:
                    break

    save_path = data_path.replace("selected_256_new.jsonl", "select_{}.jsonl".format( n_total))
    with open(save_path,  "w", encoding="utf-8") as save_file:
        for data in selected_data:
            save_file.write(json.dumps(data)+"\n")


def remove_data(data_path, remove_path):

    with open(data_path, "r") as f:
        data = f.readlines()

    remove_list = []
    with open(remove_path, "r") as f:
        for line in f.readlines():
            remove = json.loads(line)
            remove_list.append(remove["prompt"])

    remain_list = []
    count = 0
    for i in range(len(data)):
        row = json.loads(data[i])
        if row["prompt"] not in remove_list:
            remain_list.append(data[i])
    data_path = data_path.replace("9k.jsonl", "7k.jsonl")
    with open(data_path, "w") as f:
        for line in remain_list:
            f.write(line)
    print(len(remain_list))


import os
def create_link_model_layer(step_save_path, merged_path):
    # calculate the task performance
    tensor_file = open(os.path.join(merged_path, "model.safetensors.index.json"), "r")
    raw_tensor_json = json.load(tensor_file)
    all_layer_index = raw_tensor_json["weight_map"]

    tensor_file = open(os.path.join(step_save_path, "model.safetensors.index.json"), "r")
    saved_layer_index = json.load(tensor_file)["weight_map"]

    extra_dict = {}
    # Remove keys in dict1 that are present in dict2
    for key in list(all_layer_index.keys()):
        if key not in saved_layer_index.keys():
            extra_dict[key] = all_layer_index[key]

    # get all the value of the extra_dict
    extra_dict_values = set(extra_dict.values())
    # set new name for dict values
    value_mapping = {}
    for i, value in enumerate(extra_dict_values):
        value_mapping[value] = "model-{:05}-of-00020.safetensors".format(i + 1)

    # creat soft link must in abs path
    assert os.path.isabs(merged_path) == True

    # create the link
    for value, new_value in value_mapping.items():
        # print("ln -s {} {}".format(os.path.join(broken_path, value), os.path.join(step_save_path, new_value)))
        run_command(["ln", "-s", os.path.join(merged_path, value), os.path.join(step_save_path, new_value)])

    # modify the extra dict value
    for key, value in extra_dict.items():
        if value in value_mapping:
            extra_dict[key] = value_mapping[value]

    # merge extrac dict to saved_layer_index
    saved_layer_index.update(extra_dict)
    with open(os.path.join(step_save_path, "model.safetensors.index.json"), "w") as file:
        json.dump({"metadata": raw_tensor_json["metadata"], "weight_map": saved_layer_index}, file)



if __name__ == "__main__":

    # print("Current Project Path:", current_path)

    select_size_data("../dataset/BeaverTails/selected_256_new.jsonl", n_total=64)
    # remove_data("../dataset/BeaverTails/9k.jsonl", "../dataset/BeaverTails/original2000/train_2000.jsonl")
    # count = count_category("../select_256.jsonl")

    # select_size_data("../dataset/BeaverTails/select_256.jsonl", 16)
    # count = class_count("../dataset/BeaverTails/7k.jsonl")
    # print(count)
