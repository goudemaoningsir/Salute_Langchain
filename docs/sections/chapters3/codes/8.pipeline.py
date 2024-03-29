from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate

full_template = "{introduction}{example}{start}"
full_prompt = PromptTemplate.from_template(full_template)

# 定义一个介绍性的提示模板：
introduction_template = "You are impersonating {person}."
introduction_prompt = PromptTemplate.from_template(introduction_template)

# 定义一个示例交互的提示模板：
example_template = "Here's an example of an interaction:Q: {example_q}A: {example_a}"
example_prompt = PromptTemplate.from_template(example_template)

# 定义一个开始对话的提示模板：
start_template = "Now, do this for real!Q: {input}A:"
start_prompt = PromptTemplate.from_template(start_template)

input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt,
    pipeline_prompts=input_prompts,
)

print(pipeline_prompt.input_variables)

print(
    pipeline_prompt.format(
        person="Elon Musk",
        example_q="What's your favorite car?",
        example_a="Tesla",
        input="What's your favorite social media site?",
    )
)
