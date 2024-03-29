print("####################      chap1 基于长度的示例选择器     ####################")
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

# 假设任务是创建反义词的示例。
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
example_selector = LengthBasedExampleSelector(
    # 可供选择的示例。
    examples=examples,
    # 用于格式化示例的PromptTemplate。
    example_prompt=example_prompt,
    # 格式化后的示例的最大长度。
    # 长度是通过下面的get_text_length函数来衡量的。
    max_length=25,
    # 用于获取字符串长度的函数，
    # 用于确定包含哪些示例。如果没有指定，则会使用默认值。
    # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
)
dynamic_prompt = FewShotPromptTemplate(
    # 我们提供了一个ExampleSelector而不是示例。
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

print("####################      chap2 短输入     ####################")
print(dynamic_prompt.format(adjective="big"))

print("####################      chap3 长输入     ####################")
long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
print(dynamic_prompt.format(adjective=long_string))

print("####################      chap4 添加新示例     ####################")
new_example = {"input": "big", "output": "small"}
dynamic_prompt.example_selector.add_example(new_example)
print(dynamic_prompt.format(adjective="enthusiastic"))
