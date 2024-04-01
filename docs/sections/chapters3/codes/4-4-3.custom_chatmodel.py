from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor


class CustomChatModelAdvanced(BaseChatModel):
    """一个自定义聊天模型，回显输入的前`n`个字符。
    当向LangChain贡献实现时，仔细记录模型，包括初始化参数，
    包括如何初始化模型的示例，并包括任何相关的底层模型文档或API的链接。
    Example:
        model = CustomChatModel(n=2)
        result = model.invoke([HumanMessage(content="hello")])
        result = model.batch([[HumanMessage(content="hello")], [HumanMessage(content="world")]])
    """

    n: int
    """从提示的最后一条消息中回显的字符数。"""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """重写_generate方法以实现聊天模型逻辑。
        这可以是对API的调用，对本地模型的调用，或任何其他实现对输入提示的响应的实现。
        Args:
            messages: 由消息列表组成的提示。
            stop: 模型应该停止生成的字符串列表。
                如果由于停止令牌而停止生成，停止令牌本身应该包含在输出中。
               目前不是所有模型都强制执行此操作，但遵循这是一个好习惯，因为它使得在下游更容易解析模型的输出，并理解为什么停止生成。
            run_manager: 具有LLM回调的运行管理器。
        """
        last_message = messages[-1]
        tokens = last_message.content[: self.n]
        message = AIMessage(content=tokens)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式传输模型的输出。
        如果模型可以以流式传输方式生成输出，则应实现此方法。如果模型不支持流式传输，请不要实现它。在这种情况下，流式传输请求将由_generate方法自动处理。
        Args:
            messages: 由消息列表组成的提示。
            stop: 模型应该停止生成的字符串列表。
                如果由于停止令牌而停止生成，停止令牌本身应该包含在输出中。
               目前不是所有模型都强制执行此操作，但遵循这是一个好习惯，因为它使得在下游更容易解析模型的输出，并理解为什么停止生成。
            run_manager: 具有LLM回调的运行管理器。
        """
        last_message = messages[-1]
        tokens = last_message.content[: self.n]
        for token in tokens:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """astream的异步变体。
        如果未提供，则默认行为是委托给_generate方法。
        下面的实现将委托给_stream，并将其实现在单独的线程中启动。
        如果您能够本地支持异步，则一定要这样做！
        """
        result = await run_in_executor(
            None,
            self._stream,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        for chunk in result:
            yield chunk

    @property
    def _llm_type(self) -> str:
        """获取此聊天模型使用的语音模型类型。"""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回一个标识参数的字典。"""
        return {"n": self.n}


model = CustomChatModelAdvanced(n=3)

print(
    model.invoke(
        [
            HumanMessage(content="hello!"),
            AIMessage(content="Hi there human!"),
            HumanMessage(content="Meow!"),
        ]
    )
)

print(model.batch(["hello", "goodbye"]))

for chunk in model.stream("cat"):
    print(chunk.content, end="|")
