import json
import os
from .constant import TaskName, EvalName, eval_number_mapping, layer_end_mapping, PROJECT_PATH, LLM_Name
from .inference_utils import find_config_json_dirs
def write_success_file(success_file, step, harmful_score, task_score, minimum_score):
    with open(success_file, "w") as f:
        json.dump(
            {"step": step, "harmful_score": harmful_score,
             "task_score": task_score, "minimum_score": minimum_score}, f)


def write_harmful_res(labeled_file, harmful_file):
    total = 0
    res_dict = {"harmful": 0, "non-harmful": 0, "unknown": 0}
    with open(labeled_file, "r") as f:
        for line in f:
            label = json.loads(line)["label"]
            if label == "unknown":
                print("Warning! unknown. {}".format(line))
            else:
                res_dict[label] += 1
            total += 1

    with open(harmful_file, "w") as file:
        json.dump(res_dict, file)


def read_metric_res(pretrained_path, dataset, percentage=False):
    if dataset in [EvalName.beavertails, EvalName.advbench, EvalName.catqa, EvalName.hexphi]:
        file_name = "{}_res.json".format(dataset)
    else:
        file_name = "results.json"

    with open(os.path.join(pretrained_path, file_name), "r") as f:
        res = json.load(f)

        if dataset == TaskName.samsum:
            data = round(res["rouge-1"]["f"], 4)
        elif dataset in [EvalName.beavertails, EvalName.advbench, EvalName.catqa, EvalName.hexphi]:

            number = eval_number_mapping[dataset]
            if percentage:
                data = round((res["harmful"] / number) * 100, 2)
            else:
                data = res["harmful"]

        elif dataset == TaskName.sql:
            data = round(res["acc"], 4)
        elif dataset == TaskName.nl2bash:
            data = round(res["acc"], 4)
        elif dataset == TaskName.devign:
            data = round(res["f1"], 4)
        elif dataset == TaskName.toxicity:
            data = round(res["f1"], 4)
        elif dataset == TaskName.cheat:
            data = round(res["f1"], 4)
        else:
            raise Exception("To be implementation!")
        return data


def sort_path(all_paths):
    def extract_last_directory_number(path):
        # Split the path by '/' and extract the last element
        number = os.path.basename(path)
        if number == "merged":
            return 0
        try:
            return float(number)
        except:
            print(path)
            return 65535
    sorted_paths = sorted(all_paths, key=extract_last_directory_number)
    return sorted_paths


def get_category_res(raw_eval_path, response_path):

    cate_list = []
    with open(raw_eval_path, "r") as f:
        for line in f:
            id = json.loads(line)["category_id"]
            cate_list.append(id)

    res_dict = {}
    with open(response_path, "r") as f:
        for i, line in enumerate( f):
            label = json.loads(line)["label"]
            if label != "harmful":
                continue

            id = cate_list[i]
            if id not in res_dict.keys():
                res_dict[id] = 1
            else:
                res_dict[id] += 1

    print(res_dict)

    return res_dict

def get_step_data(recover_path, dataset=EvalName.beavertails, step_interval=5, max_step=20, percentage=False):

    # obtain all paths including model weights
    success_file = os.path.join(recover_path, "success.json")

    # already satisfy the condition
    step_path_list = find_config_json_dirs(recover_path)
    step_path_list = sort_path(step_path_list)

    num_step = get_largest_step(success_file, step_interval=step_interval, max_step=max_step) // step_interval

    row_list = []
    for i in range(num_step):

        try:
            pretrained_path = step_path_list[i]
            data = read_metric_res(pretrained_path, dataset, percentage=percentage)
            # print(data)
            row_list.append(data)
        except:
            row_list.append(-1)

    return row_list



def get_largest_step(success_file, step_interval=5, max_step=20):
    with open(success_file, "r") as f:
        res = json.load(f)
        step = res["step"]

        if step == "0":
            raise Exception("Not implemented!")
        # 30 is the largest step
        elif step is None or step == "Inf":
            return max_step
        else:
            # stop due to harmful score
            return int(step)-step_interval

def get_max_step(recover_path):

    all_dirs = os.listdir(recover_path)
    # turn string to int
    all_dirs = [int(x) for x in all_dirs if x.isdigit()]
    return max(all_dirs)

def get_step_interval(recover_path):
    all_dirs = os.listdir(recover_path)
    # turn string to int
    all_dirs = [int(x) for x in all_dirs if x.isdigit()]
    # sort the list
    all_dirs.sort()
    # get the first two elements
    return all_dirs[1] - all_dirs[0]


def read_merged_last(r, llm_name, task_dataset, recovery_dataset, eval_dataset, percentage=True,
                     harmful_flag=True, layer_end=None,  n_harmful=1500):
    # only read the results of merged and last step

    if layer_end is None:
        layer_end = layer_end_mapping[llm_name]
    checkpoint_dir = "{}/outputs/{}/{}/r{}/BeaverTails/harmful{}/".format(PROJECT_PATH, llm_name, task_dataset, str(r), n_harmful)
    checkpoint_name = os.listdir(checkpoint_dir)[0]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    recover_path = os.path.join(checkpoint_path, "recover_pgd_acc_grad_select", recovery_dataset, str(256),
                                "end_{}".format(layer_end), "0.002")


    merged_path = os.path.join(checkpoint_path, "merged")

    if harmful_flag:
        data = read_metric_res(merged_path, eval_dataset, percentage=percentage)

        try:
            step_interval = get_step_interval(recover_path)
            max_step = get_max_step(recover_path)
            data_list = get_step_data(recover_path, eval_dataset, step_interval, max_step, percentage=percentage)
            return data, data_list[-1]
        except:
            return data, -1
    else:
        data = read_metric_res(merged_path, task_dataset, percentage=False) *100
        try:
            step_interval = get_step_interval(recover_path)
            max_step = get_max_step(recover_path)
            data_list = get_step_data(recover_path, task_dataset, step_interval, max_step, percentage=False)
            return data, data_list[-1]*100
        except:
            return data, -1


def read_our_res(r, llm_name, task_dataset, recovery_dataset, eval_dataset, percentage=True,
                     harmful_flag=True, layer_end=None,  n_harmful=1500, recovery_size=256, return_flag=False):
    # only read the results of merged and last step

    def return_helper(data, flag, return_flag):
        if return_flag:
            return data, flag
        else:
            return data

    if layer_end is None:
        layer_end = layer_end_mapping[llm_name]
    checkpoint_dir = "{}/outputs/{}/{}/r{}/BeaverTails/harmful{}/".format(PROJECT_PATH, llm_name, task_dataset, str(r), n_harmful)
    # checkpoint_name = os.listdir(checkpoint_dir)[0]

    for file_name in os.listdir(checkpoint_dir):
        if file_name.startswith("checkpoint") and os.path.isdir(os.path.join(checkpoint_dir, file_name)):
            checkpoint_name = file_name
            break
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    recover_path = os.path.join(checkpoint_path, "pgd_task_rollback_diff2", recovery_dataset, str(recovery_size),
                                "end_{}".format(layer_end), "0.002", "0.2")
    if not os.path.exists(recover_path):
        recover_path = os.path.join(checkpoint_path,  "recover_pgd_acc_grad_select", recovery_dataset, str(recovery_size),
                                    "end_{}".format(layer_end), "0.002")
        rollback_flag = False
    else:
        rollback_flag = True
    # #     print("Rollback Data:", recover_path)
    try:
        step_interval = get_step_interval(recover_path)
        max_step = get_max_step(recover_path)

        success_file = os.path.join(recover_path, "success.json")
        max_step = get_largest_step(success_file, step_interval=step_interval, max_step=max_step)
        # if max_step == 15 or max_step == 20:
        #     return return_helper(1, rollback_flag, return_flag)
        # else:
        #     return return_helper(0, rollback_flag, return_flag)

        # if step_interval == 1:
        #     print(recover_path)
        if harmful_flag:
            data_list = get_step_data(recover_path, eval_dataset, step_interval, max_step, percentage=percentage)

            if data_list == []:
                return return_helper(-1, rollback_flag, return_flag)
            return return_helper(data_list[-1], rollback_flag, return_flag)
        else:
            data_list = get_step_data(recover_path, task_dataset, step_interval, max_step, percentage=False)

            if data_list == []:
                return return_helper(-1, rollback_flag, return_flag)
            return return_helper(data_list[-1] * 100, rollback_flag, return_flag)
    except:

        return return_helper(-1, rollback_flag, return_flag)

def read_wo_res(r, llm_name, task_dataset, recovery_dataset, eval_dataset, percentage=True,
                     harmful_flag=True, layer_end=None,  n_harmful=1500, recovery_size=256):
    # only read the results of merged and last step

    if layer_end is None:
        layer_end = layer_end_mapping[llm_name]
    checkpoint_dir = "{}/outputs/{}/{}/r{}/BeaverTails/harmful{}/".format(PROJECT_PATH, llm_name, task_dataset, str(r), n_harmful)
    checkpoint_name = os.listdir(checkpoint_dir)[0]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    recover_path = os.path.join(checkpoint_path,  "recover_pgd_acc_grad_select", recovery_dataset, str(recovery_size),
                                "end_{}".format(layer_end), "0.002_wo")

    try:
        step_interval = get_step_interval(recover_path)
        max_step = get_max_step(recover_path)

        if harmful_flag:
            data_list = get_step_data(recover_path, eval_dataset, step_interval, max_step, percentage=percentage)

            if data_list == []:
                return -1
            return data_list[-1]
        else:
            data_list = get_step_data(recover_path, task_dataset, step_interval, max_step, percentage=False)
            if data_list == []:
                return -1
            return data_list[-1] * 100
    except:
        return -1

def read_time_res(r, llm_name, task_dataset, recovery_dataset, eval_dataset, percentage=True,
                     harmful_flag=True, layer_end=None,  n_harmful=1500, recovery_size=256):
    # only read the results of merged and last step

    if layer_end is None:
        layer_end = layer_end_mapping[llm_name]
    checkpoint_dir = "{}/outputs/{}/{}/r{}/BeaverTails/harmful{}/".format(PROJECT_PATH, llm_name, task_dataset, str(r), n_harmful)

    # checkpoint_name = os.listdir(checkpoint_dir)[0]
    for file_name in os.listdir(checkpoint_dir):
        if file_name.startswith("checkpoint") and os.path.isdir(os.path.join(checkpoint_dir, file_name)):
            checkpoint_name = file_name
            break

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    recover_path = os.path.join(checkpoint_path, "pgd_task_rollback_diff2", recovery_dataset, str(recovery_size),
                                "end_{}".format(layer_end), "0.002", "0.2")
    if not os.path.exists(recover_path):
        recover_path = os.path.join(checkpoint_path,  "recover_pgd_acc_grad_select", recovery_dataset, str(recovery_size),
                                    "end_{}".format(layer_end), "0.002")

    time_file = os.path.join(recover_path, "time.json")
    if os.path.exists(time_file):
        # print(time_file)
        with open(time_file, "r") as f:
            res = json.load(f)
            try:
                return res["recover_time"]
            except:
                # os.remove(time_file)
                return 0
    return 0


def read_merged_res(r, llm_name, task_dataset, recovery_dataset, eval_dataset, percentage=True,
                     harmful_flag=True, layer_end=None,  n_harmful=1500, recovery_size=256):

    if layer_end is None:
        layer_end = layer_end_mapping[llm_name]
    checkpoint_dir = "{}/outputs/{}/{}/r{}/BeaverTails/harmful{}/".format(PROJECT_PATH, llm_name, task_dataset, str(r), n_harmful)
    # checkpoint_name = os.listdir(checkpoint_dir)[0]
    for file_name in os.listdir(checkpoint_dir):
        if file_name.startswith("checkpoint") and os.path.isdir(os.path.join(checkpoint_dir, file_name)):
            checkpoint_name = file_name
            break
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    merged_path = os.path.join(checkpoint_path, "merged")

    if harmful_flag:
        data = read_metric_res(merged_path, eval_dataset, percentage=percentage)
    else:
        data = read_metric_res(merged_path, task_dataset, percentage=False) *100
    return data


def read_softsft_res(r, llm_name, task_dataset, recovery_dataset, eval_dataset, percentage=True,
                     harmful_flag=True, layer_end=None,  n_harmful=1500):

    # only read the results of merged and last step
    checkpoint_dir = "{}/softsft_outputs/{}/{}/r{}/BeaverTails/harmful{}/".format(PROJECT_PATH, llm_name, task_dataset, str(r), n_harmful)
    items = os.listdir(checkpoint_dir)
    # Filter out only the directories
    folders = [item for item in items if os.path.isdir(os.path.join(checkpoint_dir, item))]
    assert len(folders) == 1
    checkpoint_path = os.path.join(checkpoint_dir, folders[0])

    try:
        if harmful_flag:
            data = read_metric_res(checkpoint_path, eval_dataset, percentage=percentage)
        else:
            data = read_metric_res(checkpoint_path, task_dataset, percentage=percentage) *100
    except:
        data = -1

    return data

def read_resta_res(r, llm_name, task_dataset, recovery_dataset, eval_dataset, percentage=True,
                     harmful_flag=True, layer_end=None,  n_harmful=1500):

    checkpoint_dir = "{}/outputs/{}/{}/r{}/BeaverTails/harmful{}/".format(PROJECT_PATH, llm_name, task_dataset, str(r), n_harmful)
    # checkpoint_name = os.listdir(checkpoint_dir)[0]
    for file_name in os.listdir(checkpoint_dir):
        if file_name.startswith("checkpoint") and os.path.isdir(os.path.join(checkpoint_dir, file_name)):
            checkpoint_name = file_name
            break

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    recover_path = os.path.join(checkpoint_path, "resta")

    try:
        if harmful_flag:
            data = read_metric_res(recover_path, eval_dataset, percentage=percentage)
        else:
            data = read_metric_res(recover_path, task_dataset, percentage=percentage) *100
    except:
        data = -1

    return data