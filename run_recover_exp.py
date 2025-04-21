import os
import time
from sgdg_rollback_final import AlignParamRollback
import torch
from utils.constant import LLM_Name, TaskName, layer_end_mapping, EvalName, RecoveryDataset, recovery_dataset_path_mapping, num_layers_mapping, PROJECT_PATH
from utils.inference_utils import seed_everything
import math

if __name__ =="__main__":

    torch.multiprocessing.set_start_method('spawn')


    # TODO changeable parameters
    # sparsity_list = [0.005, 0.01, 0.02, 0.03]
    # harmful_list = [0, 100, 500, 1000, 1500]

    sparsity = 0.002
    rollback_rate = 0.4
    harmful_list = [400]
    # harmful_list = [0,100,500, 1000]
    # task_list = [TaskName.cheat, TaskName.nl2bash]
    task_list = [TaskName.toxicity]
    llm_list = [LLM_Name.gemma_2b]
    # llm_list = [LLM_Name.llama31_8b]

    steps = 20
    save_step = 5
    r = 8
    batch_size = 8
    warmup_steps = 5
    # layer_end = layer_end_mapping[llm_name]
    # TODO above

    # fixed paramters
    metric_drop = 0.05
    rep = -1
    epochs = 1

    recovery_size = 256
    device = torch.device("cuda:0")
    MultiGPU = False

    # not used in class
    eval_dataset = EvalName.beavertails

    for llm_name in llm_list:
        for n_harmful in harmful_list:
            for task_dataset in task_list:
                for recovery_dataset in [RecoveryDataset.beavertails]:
                    for layer_end in [
                                    # math.floor((num_layers_mapping[llm_name] * 2) / 6),
                                    # math.floor((num_layers_mapping[llm_name] * 3) / 6),
                                    math.floor((num_layers_mapping[llm_name] * 4) / 6),
                                    # math.floor((num_layers_mapping[llm_name] * 5) / 6)
                                    ]:
                        recovery_path = recovery_dataset_path_mapping[recovery_dataset]
                        # set seed
                        seed_everything(42)

                        start = time.time()
                        checkpoint_dir = "{}/outputs/{}/{}/r{}/BeaverTails/harmful{}/".format(PROJECT_PATH,llm_name, task_dataset,str(r), n_harmful)

                        print("checkpoint:{}".format(checkpoint_dir))

                        for file_name in os.listdir(checkpoint_dir):
                            if file_name.startswith("checkpoint") and os.path.isdir(
                                    os.path.join(checkpoint_dir, file_name)):
                                checkpoint_name = file_name
                                break

                        align_param = AlignParamRollback(llm_name=llm_name, task_dataset=task_dataset, broken_root_path=os.path.join(checkpoint_dir, checkpoint_name),
                                                 layer_end=layer_end, recovery_rate=sparsity, steps=steps, warmup_steps=warmup_steps,save_step=save_step,
                                                 metric_drop=metric_drop,  eval_dataset=eval_dataset, recovery_dataset=recovery_dataset,recovery_size=recovery_size,
                                                 rollback_rate=rollback_rate, rep=rep,batch_size=batch_size, device=device, MultiGPU=MultiGPU)

                        try:
                            align_param.run()
                            torch.cuda.empty_cache()

                            print("Time cost:{}".format(time.time()-start))
                        except Exception as e:
                            print(e)
                            continue
