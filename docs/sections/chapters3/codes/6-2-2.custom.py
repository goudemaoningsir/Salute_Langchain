from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

# [bool] 描述了泛型的一个参数化。
# 它基本上指示了解析的返回类型是什么
# 在这种情况下，返回类型是True或False


class BooleanOutputParser(BaseOutputParser[bool]):
    """自定义布尔解析器。"""

    true_val: str = "YES"
    false_val: str = "NO"

    def parse(self, text: str) -> bool:
        cleaned_text = text.strip().upper()
        if cleaned_text not in (self.true_val.upper(), self.false_val.upper()):
            raise OutputParserException(
                f"BooleanOutputParser expected output value to either be "
                f'"{self.true_val} or {self.false_val} (case-insensitive). '
                f"Received {cleaned_text}."
            )
        return cleaned_text == self.true_val.upper()

    @property
    def _type(self) -> str:
        return "boolean_output_parser"


parser = BooleanOutputParser()
print(parser.invoke("YES"))

try:
    parser.invoke("MEOW")
except Exception as e:
    print(f"Triggered an exception of type: {type(e)}")

parser = BooleanOutputParser(true_val="OKAY")
print(parser.invoke("OKAY"))
