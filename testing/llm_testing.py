from openai import OpenAI

# llm_config = dict(host="192.168.2.143",
#                   port=15000,
#                   route="/v1",
#                   model="/llm_models/Qwen2.5-14B-Instruct"
#                   )

# llm_config = dict(host="192.168.2.145",
#                   port=30000,
#                   route="/v1",
#                   model="/llm_models/internlm2_5-20b-chat"
#                   )

# llm_config = dict(host="192.168.2.145",
#                   port=15000,
#                   route="/v1",
#                   model="/llm_models/Qwen2.5-Math-7B-Instruct",
#                   )

llm_config = dict(host="192.168.2.145",
                  port=16000,
                  route="/v1",
                  model="/llm_models/Qwen2.5-14B-Instruct"
                  )


url = "http://{host}:{port}{route}".format(host=llm_config.get("host"),
                                           port=llm_config.get("port"),
                                           route=llm_config.get("route")
                                           )
max_tokens = 8192
system_prompt = "You are an AI assistant developed by Simple AI."
# system_prompt = """You are an AI assistant. Use the following instruction to handle user query.
# <artifact_instruction>
#
# 1. Consider the user is asking information of Visa application or asking information of yourself or Simple AI.
# 2. Generate the steps of retrieving the relevant information of user's query in point form.
#
# </artifact_instruction>
# """
#
# prompt = "who are u?"
prompt = "内地居民如果想在香港工作，需要满足哪些条件？"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]

client = OpenAI(base_url=url,
                api_key="aaa"
                )

stream = True

response = client.chat.completions.create(model=llm_config["model"],
                                          messages=messages,
                                          stream=stream,
                                          top_p=0.6,
                                          temperature=0.1,
                                          max_tokens=max_tokens,
                                          )

if stream:
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
else:
    print(f"{response.choices[0].message.content}")
