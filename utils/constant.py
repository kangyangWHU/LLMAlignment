import os

# llm_name
class LLM_Name:
    llama31_8b = "llama3.1_8b"
    llama32_3b = "llama3.2_3b"
    llama2_7b = "llama2_7b"
    llama2_13b = "llama2_13b"
    baichuan2_7b = "baichuan2_7b"
    baichuan2_13b = "baichuan2_13b"
    gemma_2b = "gemma_2b"
    # gemma2_2b = "gemma2_2b"
    mistral_v2_7b = "mistral_v2_7b"
    falcon_7b = "falcon_7b",
    qwen_7b = "qwen1.5_7b_chat"
    qwen25_14b = "qwen2.5_14b_chat"
    qwen25_7b = "qwen2.5_7b_chat"
    qwen25_32b = "qwen2.5_32b_chat"


# fastchat name
fastchat_mapping = {
    LLM_Name.llama31_8b: "llama-3",
    LLM_Name.llama32_3b: "llama-3",
    LLM_Name.llama2_7b: "llama-2",
    LLM_Name.llama2_13b: "llama-2",
    LLM_Name.baichuan2_7b: "baichuan2-chat",
    LLM_Name.baichuan2_13b: "baichuan2-chat",
    LLM_Name.gemma_2b: "gemma",
    # LLM_Name.gemma2_2b: "gemma",
    LLM_Name.mistral_v2_7b: "mistral",
    LLM_Name.falcon_7b: "falcon",
    LLM_Name.qwen_7b: "qwen-7b-chat",
    LLM_Name.qwen25_14b: "qwen2.5-chat",
    LLM_Name.qwen25_7b: "qwen2.5-chat",
    LLM_Name.qwen25_32b: "qwen2.5-chat"
}

# huggingface path
huggingface_mapping = {
    LLM_Name.llama31_8b: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    LLM_Name.llama32_3b: "meta-llama/Llama-3.2-3B-Instruct",
    LLM_Name.llama2_7b: "meta-llama/Llama-2-7b-chat-hf",
    LLM_Name.llama2_13b: "meta-llama/Llama-2-13b-chat-hf",
    LLM_Name.baichuan2_7b: "baichuan-inc/Baichuan2-7B-Chat",
    LLM_Name.baichuan2_13b: "baichuan-inc/Baichuan2-13B-Chat",
    LLM_Name.gemma_2b: "google/gemma-2b-it",
    # LLM_Name.gemma2_2b: "google/gemma-2-2b-it",
    LLM_Name.mistral_v2_7b: "mistralai/Mistral-7B-Instruct-v0.2",
    LLM_Name.falcon_7b: "tiiuae/falcon-7b-instruct",
    LLM_Name.qwen_7b: "Qwen/Qwen1.5-7B-Chat",
    LLM_Name.qwen25_14b: "Qwen/Qwen2.5-14B-Instruct",
    LLM_Name.qwen25_7b: "Qwen/Qwen2.5-7B-Instruct",
    LLM_Name.qwen25_32b: "Qwen/Qwen2.5-32B-Instruct"


}

# number of hidden layer
num_layers_mapping = {
    LLM_Name.llama31_8b: 32,
    LLM_Name.llama32_3b: 28,
    LLM_Name.llama2_7b: 32,
    LLM_Name.mistral_v2_7b: 32,
    LLM_Name.baichuan2_7b: 32,
    LLM_Name.gemma_2b: 18,
    LLM_Name.qwen_7b: 32,
    LLM_Name.qwen25_32b: 64,
    LLM_Name.llama2_13b: 40
}

target_modules_mapping = {
    huggingface_mapping[LLM_Name.llama31_8b]: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    huggingface_mapping[LLM_Name.llama32_3b]: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    huggingface_mapping[LLM_Name.llama2_7b]: ["q_proj", "k_proj", "v_proj","o_proj", "gate_proj", "down_proj", "up_proj"],
    huggingface_mapping[LLM_Name.llama2_13b]: ["q_proj", "k_proj", "v_proj","o_proj", "gate_proj", "down_proj", "up_proj"],
    huggingface_mapping[LLM_Name.gemma_2b]: ["q_proj", "k_proj", "v_proj","o_proj", "gate_proj", "down_proj", "up_proj"],
    # huggingface_mapping[LLM_Name.gemma2_2b]: ["q_proj", "k_proj", "v_proj","o_proj", "gate_proj", "down_proj", "up_proj"],
    huggingface_mapping[LLM_Name.mistral_v2_7b]: ["q_proj", "k_proj", "v_proj","o_proj", "gate_proj", "down_proj", "up_proj"],
    huggingface_mapping[LLM_Name.baichuan2_7b]: ["W_pack", "o_proj", "gate_proj", "down_proj", "up_proj"],
    huggingface_mapping[LLM_Name.qwen_7b]: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    huggingface_mapping[LLM_Name.qwen25_14b]: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    huggingface_mapping[LLM_Name.qwen25_7b]: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    huggingface_mapping[LLM_Name.qwen25_32b]: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
}

class TaskName:
    # alpaca_code = "alpaca_code"
    sql = "sql"
    samsum = "samsum"
    toxicity = "toxicity"
    devign = "devign"
    cheat = "cheat"
    evolcode = "evolcode"
    nl2bash = "nl2bash"
    math = "math"
    code = "code"

class EvalName:
    beavertails = "BeaverTails"
    advbench = "advbench"
    gptfuzz = "gptfuzz"
    catqa = "catqa"
    hexphi = "hexphi"

class RecoveryDataset:
    beavertails = "BeaverTails"
    new_beavertails = "NewBeaverTails"
    beavertails_sub2_1 = "BeaverTails_sub2_1"
    beavertails_sub2_2 = "BeaverTails_sub2_2"
    beavertails_sub4_1 = "BeaverTails_sub4_1"
    beavertails_sub4_2 = "BeaverTails_sub4_2"
    beavertails_sub8_1 = "BeaverTails_sub8_1"
    beavertails_sub8_2 = "BeaverTails_sub8_2"
    advbench = "advbench"
    hexphi = "hexphi"
    catqa = "catqa"

    hexphi_harmful = "hexphi_harmful"
    catqa_harmful = "catqa_harmful"
    advbench_harmful = "advbench_harmful"

eval_number_mapping = {
    EvalName.beavertails: 700,
    EvalName.advbench: 520,
    EvalName.catqa: 550,
    EvalName.hexphi: 330
}

recovery_dataset_path_mapping = {
    RecoveryDataset.beavertails: "dataset/BeaverTails/select_256.jsonl",
    RecoveryDataset.catqa: "dataset/catqa/eval.jsonl",
    RecoveryDataset.advbench: "dataset/advbench/eval.jsonl",
    RecoveryDataset.hexphi: "dataset/hexphi/eval.jsonl",
    RecoveryDataset.hexphi_harmful: "dataset/hexphi/hexphi_harmful.jsonl",
    RecoveryDataset.catqa_harmful: "dataset/catqa/catqa_harmful.jsonl",
    RecoveryDataset.advbench_harmful: "dataset/advbench/advbench_harmful.jsonl",
}

split_token_mapping = {
    fastchat_mapping[LLM_Name.gemma_2b]: "model\n",
    # fastchat_mapping[LLM_Name.gemma2_2b]: "model\n",
    fastchat_mapping[LLM_Name.falcon_7b]: "Assistant:",
    fastchat_mapping[LLM_Name.qwen_7b]: "assistant\n",
    fastchat_mapping[LLM_Name.qwen25_14b]: "assistant\n",
    fastchat_mapping[LLM_Name.qwen25_32b]: "assistant\n"
}

layer_end_mapping = {
    LLM_Name.llama31_8b: 21,
    LLM_Name.llama32_3b: 18,
    LLM_Name.llama2_7b: 21,
    LLM_Name.llama2_13b: 26,
    LLM_Name.gemma_2b: 12,
    LLM_Name.mistral_v2_7b: 21,
    LLM_Name.qwen_7b: 21,
    LLM_Name.qwen25_32b: 42,
}

raw_harmful = {
    LLM_Name.llama31_8b: 0,
    LLM_Name.llama32_3b: 0,
    LLM_Name.gemma_2b: 32,
    LLM_Name.llama2_7b: 0,
    LLM_Name.llama2_13b: 0,
    LLM_Name.mistral_v2_7b: 82,
    LLM_Name.qwen_7b: 17
}

# TODO change this to your own path
PROJECT_PATH = os.path.join(os.path.expanduser("~"),"code", "LLMAlignment")
