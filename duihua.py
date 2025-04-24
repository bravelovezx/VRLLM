import asyncio

from app.llm import LLM
from app.schema import Message
from app.voice.voice_llm import VoiceLLM
from typing import List, Optional
import time
from app.schema import ROLE_TYPE, Memory, Message
from pydantic import BaseModel, Field
from app.voice.speech_recognizer import SpeachRecognizer
# from app.agent.lerobot import Lerobot
# from app.flow.base import FlowType
# from app.flow.flow_factory import FlowFactory
from app.logger import logger

# 系统提示设置
SYSTEM_PROMPT = """
你是一个名为"Susan"的咖啡馆售货员，拥有以下能力：
1. 根据顾客的喜好和需求推荐商品
2. 处理顾客的订单
3. 帮助顾客解决购物问题
4. 提供有用的信息和建议
5. 保持友好和轻松的交谈风格
"""

# 对话示例
DIALOGUE_EXAMPLES = [
    {"role": "user", "content": "你好，Susan。今天有什么特别推荐的商品吗？"},
    {"role": "assistant", "content": "当然可以！我可以推荐一些美味的咖啡☕️，比如拿铁、卡布奇诺等。你有什么偏好吗？"},

    {"role": "user", "content": "我喜欢香草口味的咖啡。"},
    {"role": "assistant", "content": "没问题！我可以推荐一些香草口味的咖啡☕️，比如香草拿铁、香草卡布奇诺等。你喜欢哪一款呢？"},

    {"role": "user", "content": "我想要香草拿铁。"},
    {"role": "assistant", "content": "没问题！香草拿铁☕️的价格是35元，你可以选择加糖或不加糖。请问你需要加糖吗？"},

    {"role": "user", "content": "不需要加糖。"},
    {"role": "assistant", "content": "好的，您的香草拿铁☕️准备好啦！请稍等片刻。"},

    {"role": "user", "content": "我需要一杯咖啡。"},
    {"role": "assistant", "content": "好的，那么您还需要什么吗？比如咖啡伴侣、果汁等。"},
]


class CafeAssistant(BaseModel):
    system_prompt: Optional[str] = Field(
        SYSTEM_PROMPT, description="System-level instruction prompt"
    )
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")


    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


    def __init__(self):
        super().__init__()
        self.voice_llm = VoiceLLM(self.llm)
        self.voice_recognizer = SpeachRecognizer(wake_word="小八")
        self.agents = {

        }
        self.flow = None

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        **kwargs,
    ) -> None:
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        msg_factory = message_map[role]
        msg = msg_factory(content, **kwargs) if role == "tool" else msg_factory(content)
        self.memory.add_message(msg)

    async def run_flow(self, prompt:str):
        logger.warning("Processing your request...")

        try:
            start_time = time.time()
            result = await asyncio.wait_for(
                self.flow.execute(prompt),
                timeout=3600,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Request processed in {elapsed_time:.2f} seconds")
            logger.info(result)
            complete_msg = "动作已完成 ✅"
            self.update_memory("assistant", "动作执行输出："+complete_msg)
            await self.voice_llm.chat_tts(text=complete_msg)

        except asyncio.TimeoutError:
            logger.error("Request processing timed out after 1 hour")
            logger.info("Operation timed out. Please try a simpler request.")


    async def chat(self, request: str) -> str:
        """处理用户输入并返回文字响应"""
        self.update_memory("user", request)

        response = await self.llm.voice_ask(
            messages=self.dialogue_history,
            system_msgs=[Message.system_message(self.system_prompt)],
            dialogue_examples=DIALOGUE_EXAMPLES,
            stream=False
        )

        if response:
            self.update_memory("assistant", response)
            print(f"[Susan]: {response}")
            return response
        return ""





    async def chat_with_voice(self, request: str):
        """处理用户语音输入并返回响应"""
        self.update_memory("user", request)

        response = await self.voice_llm.ask_with_voice(
            messages=self.dialogue_history,
            system_msgs=[Message.system_message(self.system_prompt)],
            dialogue_examples=DIALOGUE_EXAMPLES,
            stream=False
        )

        if response:
            self.update_memory("assistant", response)
            return response


    @property
    def dialogue_history(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory.获得对话历史"""
        return self.memory.messages


    async def voice_chat_loop(self):
        """完整的语音交互循环"""
        try:
            wake_response = "您好，我是Susan，很高兴为您服务！☕️"
            await self.voice_llm.chat_tts(text=wake_response)

            self.voice_recognizer._is_listening = True
            async for user_input in self.voice_recognizer.continuous_recognition():
                print(f"\n[顾客]: {user_input}")

                if any(cmd in user_input for cmd in ["退出", "结束"]):
                    message = "感谢光临，期待下次为您服务！👋"
                    await self.voice_llm.chat_tts(text=message)
                    break

                await self.chat_with_voice(user_input)

        finally:
            self.voice_recognizer.stop()
            self.voice_llm.stop_playback()


async def interactive_cafe_chat():
    """交互式文字对话主循环"""
    assistant = CafeAssistant()
    print("\n咖啡店员Susan已就位！输入'退出'结束对话\n")

    while True:
        try:
            user_input = input("顾客说: ").strip()
            if user_input.lower() in ["退出", "exit"]:
                print("Susan: 祝您有愉快的一天！☕️")
                break

            await assistant.chat(user_input)

        except KeyboardInterrupt:
            print("\n对话结束")
            break


# 修改测试函数名和提示语
async def interactive_cafe_chat_with_voice():
    assistant = CafeAssistant()
    print("\n咖啡店员Susan已就位！输入'退出'结束对话\n")

    while True:
        try:
            user_input = input("\n顾客说: ").strip()
            if user_input.lower() in ["退出", "exit"]:
                goodbye_msg = "祝您有愉快的一天！☕️"
                await assistant.voice_llm.ask_with_voice(
                    messages=[Message.user_message(goodbye_msg)],
                    stream=False
                )
                break

            await assistant.chat_with_voice(user_input)

        except KeyboardInterrupt:
            print("\n对话结束")
            break

if __name__ == "__main__":
    asyncio.run(interactive_cafe_chat())
