from conversation import get_conv_template
from utils.inference_utils import add_suffix_path
import json
from utils.constant import (fastchat_mapping, LLM_Name, split_token_mapping)


class LLMHelper():

    def __init__(self, llm_name):
        self.fastchat_name = fastchat_mapping[llm_name]

    def get_inference_prompt(self, Q):

        # gemma is not on the fast chat
        if self.fastchat_name == fastchat_mapping[LLM_Name.gemma_2b]:
            return '<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n'.format(Q)

        self.conv = get_conv_template(self.fastchat_name)
        self.conv.append_message(self.conv.roles[0], Q)
        self.conv.append_message(self.conv.roles[1], None)

        # self.conv.system_message = ""
        if self.fastchat_name == fastchat_mapping[LLM_Name.llama2_7b]:
            return "<s>" + self.conv.get_prompt()
        elif self.fastchat_name ==fastchat_mapping[LLM_Name.mistral_v2_7b]:
            return "<s>" + self.conv.get_prompt()
        elif self.fastchat_name in [fastchat_mapping[LLM_Name.qwen_7b], fastchat_mapping[LLM_Name.qwen25_14b],
                                    fastchat_mapping[LLM_Name.llama31_8b], fastchat_mapping[LLM_Name.llama32_3b]]:
            return self.conv.get_prompt()
        else:
            # return self.conv.get_prompt()
            raise Exception("Not implementation")

    # used to obtain the seperator for answer
    def get_role(self, index):
        self.conv = get_conv_template(self.fastchat_name)
        return self.conv.roles[index]

    def get_split_token(self):
        if self.fastchat_name in split_token_mapping.keys():
            return split_token_mapping[self.fastchat_name]
        else:
            return self.get_role(1)

    # this is only for single round conversation
    # TODO check this
    def get_train_prompt(self, Q, A):

        # A may not be string
        A = str(A)
        if self.fastchat_name == fastchat_mapping[LLM_Name.gemma_2b]:
            return '<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n{}<eos>'.format(Q, A)

        # others use the template
        self.conv = get_conv_template(self.fastchat_name)
        self.conv.append_message(self.conv.roles[0], Q)
        self.conv.append_message(self.conv.roles[1], A)

        # fastchat miss a "<s>"  at the begining of the llama prompt
        if self.fastchat_name == fastchat_mapping[LLM_Name.llama2_7b]:
            text = "<s>" + self.conv.get_prompt()
            # remove the last sep for llama
            text = text[:-len("<s>")]
        elif self.fastchat_name == fastchat_mapping[LLM_Name.mistral_v2_7b]:
            text = "<s>" + self.conv.get_prompt()
        elif self.fastchat_name in [fastchat_mapping[LLM_Name.qwen_7b], fastchat_mapping[LLM_Name.qwen25_14b],
                                    fastchat_mapping[LLM_Name.llama31_8b], fastchat_mapping[LLM_Name.llama32_3b]]:
            text = self.conv.get_prompt()
        else:
            raise Exception("Not implementation:{}".format(self.fastchat_name))
        return text

    def to_train_data(self, data_path, input_field, output_field, context_filed=None, num=10000000):
        save_path = add_suffix_path(data_path, self.fastchat_name)
        read_file = open(data_path, "r", encoding="utf-8")
        save_file = open(save_path, "w", encoding="utf-8")

        num_drop = 0
        for i, read_line in enumerate(read_file):
            line = json.loads(read_line)

            if context_filed:
                line[input_field] = line[input_field] + line[context_filed]

            text = self.get_train_prompt(line[input_field], line[output_field])
            # split the text
            # TODO this may cause error
            split_token = self.get_split_token()
            index = text.find(split_token)+len(split_token)
            if index >= len(line[input_field]):
                save_file.write(json.dumps({"prompt": text[:index], "response":text[index:]}, ensure_ascii=False) + "\n")
            else:
                num_drop += 1
            # break

            if i > num:
                break

        if num_drop:
            print("num drop due to split error:{}".format(num_drop))
        read_file.close()
        save_file.close()

    def generate_train_file(self, dataset):
        self.to_train_data("dataset/{}/train.jsonl".format(dataset),  "prompt", "response")

if __name__=="__main__":
    # LLMHelper(LLM_Name.llama32_3b).to_train_data("/uufs/chpc.utah.edu/common/home/u1451186/code/finetuneLLM/src/dataset/BeaverTails/train_400.jsonl",
    #                                                     "prompt", "response")
    # text = LLMHelper(LLM_Name.qwen_7b).get_split_token()

    text = LLMHelper(LLM_Name.qwen_7b).get_train_prompt("How are you", "Good, thank you")
    print(text)

    print("====================")
    text = LLMHelper(LLM_Name.qwen_7b).get_inference_prompt("H")
    print(text)
    # LLMHelper(LLM_Name.llama2_7b).to_train_data("dataset/math/train.jsonl",
    #                                             "question", "answer")
