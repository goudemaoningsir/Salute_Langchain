print("####################      chap1 示例选择器     ####################")
from langchain_core.example_selectors.base import BaseExampleSelector


class CustomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        # 这假设输入的一部分将是 'text' 键
        new_word = input_variables["input"]
        new_word_length = len(new_word)

        # 初始化变量以存储最佳匹配及其长度差异
        best_match = None
        smallest_diff = float("inf")

        # 遍历每个例子
        for example in self.examples:
            # 计算例子的第一个单词与新单词的长度差异
            current_diff = abs(len(example["input"]) - new_word_length)

            # 如果当前差异更接近，则更新最佳匹配
            if current_diff < smallest_diff:
                smallest_diff = current_diff
                best_match = example

        return [best_match]


examples = [
    {"input": "hi", "output": "你好"},
    {"input": "bye", "output": "再见"},
    {"input": "soccer", "output": "足球"},
]

example_selector = CustomExampleSelector(examples)

print(
    example_selector.select_examples({"input": "okay"})
)  # [{'input': 'bye', 'output': '再见'}]


example_selector.add_example({"input": "hand", "output": "手"})

selected_example = example_selector.select_examples({"input": "hand"})
print(selected_example)  # 预期输出：[{'input': 'hand', 'output': '手'}]

print("####################      chap2 示例选择器 - 使用   ####################")
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

example_prompt = PromptTemplate.from_template("Input: {input} -> Output: {output}")

prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Input: {input} -> Output:",
    prefix="Translate the following words from English to chinese:",
    input_variables=["input"],
)

print(prompt.format(input="word"))

# Translate the following words from Chinese to English:
# Input: 手 -> Output: hand
# Input: 词 -> Output:
