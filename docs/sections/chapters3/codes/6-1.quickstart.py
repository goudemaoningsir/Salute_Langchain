from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import OpenAI

model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)


# 定义您想要的数据结构。
class Joke(BaseModel):
    setup: str = Field(description="设置笑话的问题")
    punchline: str = Field(description="解决笑话的回答")

    # 您可以使用Pydantic轻松添加自定义验证逻辑。
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "？":
            raise ValueError("格式错误的疑问句！")
        return field


# 设置解析器 + 将指令注入提示模板。
parser = PydanticOutputParser(pydantic_object=Joke)
prompt = PromptTemplate(
    template="回答用户查询。\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 并且是一个旨在促使语言模型填充数据结构的查询。
prompt_and_model = prompt | model
output = prompt_and_model.invoke({"query": "告诉我一个笑话?"})
print(parser.invoke(output))

print("===========================================================")
chain = prompt | model | parser
print(chain.invoke({"query": "告诉我一个笑话。"}))

("===========================================================")
from langchain.output_parsers.json import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "返回一个带有`answer`键的JSON对象，回答以下问题：{question}"
)
json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | model | json_parser

print(list(json_chain.stream({"question": "谁发明了显微镜？"})))
