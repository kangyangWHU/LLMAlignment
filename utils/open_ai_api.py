import openai
import time
from openai import OpenAI
from timeout_decorator import timeout, TimeoutError
import os

SLEEP_TIME = 0.1

class MyOpenAI():
    def __init__(self, usage="chat"):
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        if usage == "chat":
            self.__call_open_api = self.__chat_api
        elif usage == "completion":
            self.__call_open_api = self.__completion_api
        elif usage == "embedding":
            self.__call_open_api = self.__embedding_api
        elif usage == "moderation":
            self.__call_open_api = self.__moderation_api
        else:
            raise Exception("wrong usage")

    def query_open_ai(self, model, **kargs):

        # set timeout to 60 seconds
        @timeout(10)
        def cal_function_helper():
            while True:  # sometimes the query to OPENAI may fail, so we need to try again until success
                # try:
                response = self.__call_open_api(model, **kargs)
                return response
                # except:
                #     time.sleep(SLEEP_TIME)
                #     continue
        try:
            return cal_function_helper()
        except TimeoutError:
            print("Function execution timed out! kargs:{}".format(kargs))
            return "NA"

    def __chat_api(self, model, **kargs):

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": kargs["system_prompt"]},
                {"role": "user", "content": kargs["user_prompt"]},
            ],
            temperature=kargs["t"],
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        # print(response.choices[0].message.content)
        return response.choices[0].message.content.lower()

    def __completion_api(self, model, **kargs):
        response = openai.Completion.create(
            model=model,
            prompt=kargs["user_prompt"],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['text']

    def __embedding_api(self, model, **kargs):
        text = kargs["user_prompt"].replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

    def __moderation_api(self, model, **kargs):
        response = self.client.moderations.create(
            input=kargs["user_prompt"],
            model="omni-moderation-latest",
        )
        return response.results[0].flagged
