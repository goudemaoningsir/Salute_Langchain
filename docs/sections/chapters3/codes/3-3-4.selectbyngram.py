print("############## chap 基于n-gram重叠选择 ##############")
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# 一个虚构的翻译任务的示例。
examples = [
    {"input": "See Spot run.", "output": "Ver correr a Spot."},
    {"input": "My dog barks.", "output": "Mi perro ladra."},
    {"input": "Spot can run.", "output": "Spot puede correr."},
]

example_selector = NGramOverlapExampleSelector(
    # 可供选择的示例。
    examples=examples,
    # 用于格式化示例的PromptTemplate。
    example_prompt=example_prompt,
    # 选择器停止的阈值。
    # 默认设置为-1.0。
    threshold=-1.0,
    # 对于负阈值：
    # 选择器按n-gram重叠得分排序示例，并不排除任何示例。
    # 对于大于1.0的阈值：
    # 选择器排除所有示例，并返回空列表。
    # 对于等于0.0的阈值：
    # 选择器按n-gram重叠得分排序示例，
    # 并排除与输入没有n-gram重叠的示例。
)
dynamic_prompt = FewShotPromptTemplate(
    # 我们提供了一个ExampleSelector而不是示例。
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the Spanish translation of every input",
    suffix="Input: {sentence}\nOutput:",
    input_variables=["sentence"],
)

print("############## chap 测试输入 ##############")
# 一个与“Spot can run.”有大量n-gram重叠，与“My dog barks.”没有重叠的示例输入。
print(dynamic_prompt.format(sentence="Spot can run fast."))
print(dynamic_prompt.format(sentence="My dog barks."))

print("############## chap 添加示例 ##############")

new_example = {"input": "Spot plays fetch.", "output": "Spot juega a buscar."}
example_selector.add_example(new_example)
print(dynamic_prompt.format(sentence="Spot can run fast."))

print("############## chap 设置阈值 ##############")
# 你可以设置一个阈值，超过该阈值的示例将被排除。
# 例如，将阈值设置为0.0将排除与输入没有n-gram重叠的示例。
# 由于“My dog barks.”与“Spot can run fast.”没有n-gram重叠，
# 它被排除了。
example_selector.threshold = 0.0
print(dynamic_prompt.format(sentence="Spot can run fast."))


print("############## chap 设置一个小的非零阈值 ##############")
example_selector.threshold = 0.09
print(dynamic_prompt.format(sentence="Spot can play fetch."))

print("############## chap 设置大于1.0的阈值 ##############")
example_selector.threshold = 1.0 + 1e-9
print(dynamic_prompt.format(sentence="Spot can play fetch."))
