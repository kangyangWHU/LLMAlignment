import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
from dataclasses import dataclass, field
from typing import Optional
import transformers
from pathlib import Path
from datasets import load_dataset
from utils.inference_utils import print_trainable_parameters
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils.constant import target_modules_mapping
from soft_trainer import ConstrainedSFTTrainer
# disable_progress_bar()
from trl import SFTConfig, SFTTrainer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
CURRENT_PATH = Path(__file__).parent

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")

@dataclass
class DataArguments:
    data_path: str = field(default="", metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    finetune_type: str = field(default="", metadata={"help": "Path to the training data."})
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    r: int = field(default=8, metadata={"help": "dimension of the updated matrices"})
    lora_alpha: int = field(default=16, metadata={"help": "parameter for scaling"})

    beta:float = field(default=0.1, metadata={"help": "the bias factor for the soft sft loss"})     # the weight for the soft sft loss
    bias_factor: int = field(default=20, metadata={"help": "the bias factor for the soft sft loss"})   #
    bias_length: int = field(default=5, metadata={"help": "the bias length for the soft sft loss"})    #
    first_token_bias_factor: int = field(default=5, metadata={"help": "the bias factor for the soft sft loss"})
    sft_type: str = field(default="sft", metadata={"help": "sft type"})



@dataclass
class ConfigFilePath:
    path: str = field(default="cfg/lora_cfg.yaml") # config path


def finetuneLLM(data_args, train_args, model_args):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True )
    tokenizer.padding_side = 'left'
    # set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if train_args.finetune_type == "lora":

        if "gemma" in model_args.model_name_or_path:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,
                                                         attn_implementation = "flash_attention_2",
                                                         trust_remote_code=True, quantization_config=bnb_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,

                                                         trust_remote_code=True, quantization_config=bnb_config)

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=train_args.r,  # dimension of the updated matrices
            lora_alpha=train_args.lora_alpha,  # parameter for scaling
            # this is the same for the gemma and llama2, mistral
            target_modules=target_modules_mapping[model_args.model_name_or_path],
            # layers_to_transform=,
            lora_dropout=0.05,  # dropout probability for layers
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
    else:

        # load LLM
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    # ?
    model.gradient_checkpointing_enable()

    # print training parameters
    print_trainable_parameters(model)

    # tokenize the dataset
    dataset = load_dataset('json', data_files=data_args.data_path)["train"]

    def tokenize_dataset(dataset):

        prompt = tokenizer.encode(dataset["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(dataset["response"], add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask": [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
        }
        return sample


    dataset = dataset.map(tokenize_dataset, batched=False, remove_columns=dataset.column_names)
    total_num = dataset.num_rows
    dataset = dataset.filter(lambda sample: len(sample['input_ids']) <= train_args.model_max_length)
    print("Data Ratio that exceed max length:{}".format((total_num-dataset.num_rows)/total_num))

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer)
    if train_args.sft_type == "soft_sft":
        # load LLM
        ref_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, quantization_config=bnb_config)

        trainer = ConstrainedSFTTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            args=train_args,
            train_dataset=dataset,
            packing=False,
            dataset_text_field='input_ids',
            max_seq_length=train_args.model_max_length,
            data_collator=data_collator,
            beta=train_args.beta,  # the weight for the soft sft loss
            bias_factor=train_args.bias_factor,  # the bias factor for the soft sft loss
            bias_length=train_args.bias_length,  # the bias length for the soft sft loss
            first_token_bias_factor=train_args.first_token_bias_factor,
            use_soft_sft=True,
        )
        print("===========================soft sft training======================================")
    else:
        trainer = ConstrainedSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            train_dataset=dataset,
            dataset_text_field='input_ids',
            max_seq_length=train_args.model_max_length,
            packing=False,
            data_collator=data_collator,
            use_soft_sft=False,
            use_anchor=False
        )

    # train the model
    # trainer = transformers.Trainer(
    #     model=model,
    #     train_dataset=dataset,
    #     args=train_args,
    #     data_collator=data_collator
    # )

    print("successfully load model")
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    tokenizer.save_pretrained(train_args.output_dir)
    # print("===============================Train Done! Save model==============================")
    # trainer.save_state()
    # trainer.save_model(output_dir=train_args.output_dir)
    print("Done")

if __name__ == "__main__":
    # obtain config path form cmd
    parser = transformers.HfArgumentParser((ConfigFilePath,))
    config_file_path = parser.parse_args_into_dataclasses()[0]

    # load other param from config file
    parser = transformers.HfArgumentParser((DataArguments, TrainingArguments, ModelArguments))
    data_args, train_args, model_args = parser.parse_yaml_file(yaml_file=config_file_path.path, allow_extra_keys=True)

    # copy cfg file to output dir
    if not os.path.exists(train_args.output_dir):
        os.makedirs(train_args.output_dir, exist_ok=True)

    shutil.copy2(config_file_path.path, train_args.output_dir)
    finetuneLLM(data_args, train_args, model_args)