from langchain_openai import OpenAI

print("####################      step0 计算函数执行时间     ####################")
import time


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录函数开始执行的时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录函数结束执行的时间
        elapsed_time = end_time - start_time  # 计算执行时间
        print(f"{func.__name__} took {elapsed_time} seconds to execute.")
        return result

    return wrapper


print("####################      step1 初始化模型     ####################")
# 为了使缓存效果非常明显，我们使用一个较慢的模型。
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", n=2, best_of=2)


print("####################      step2  内存缓存​     ####################")
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache

set_llm_cache(InMemoryCache())
# 第一次，它尚未缓存，所以应该需要更长时间


@timing_decorator
def predict_and_time(chat_instance, text):
    return chat_instance.invoke(text)


print(predict_and_time(llm, "Tell me a joke"))
print(predict_and_time(llm, "Tell me a joke"))
