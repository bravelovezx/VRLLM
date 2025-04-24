from typing import Optional, List, Union
from app.logger import logger
from app.llm import LLM
from app.exceptions import TokenLimitExceeded
from openai import OpenAIError
from app.schema import Message
import torch
from ChatTTS import ChatTTS
import torchaudio
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import asyncio
import tempfile
import subprocess
import sys

class AudioManager:
    """异步音频生命周期管理"""
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._active_processes = set()

    async def play_temp_audio(self, wave_data: np.ndarray) -> str:
        try:
            # 创建临时文件（自动生成唯一路径）
            temp_path = tempfile.mktemp(suffix='.wav')

            # 并行执行保存和播放
            await asyncio.gather(
                self._async_save_audio(wave_data, temp_path),
                self._async_play_with_cleanup(temp_path)
            )

            return temp_path

        except Exception as e:
            print(f"音频播放失败: {str(e)}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    async def _async_save_audio(self, wave_data: np.ndarray, path: str):
        """异步保存音频文件"""
        await asyncio.get_event_loop().run_in_executor(
            self._executor,
            lambda: torchaudio.save(path, torch.from_numpy(wave_data), 24000)
        )

    async def _async_play_with_cleanup(self, file_path: str):
        """异步播放并自动清理"""
        def _play_and_monitor():
            # 跨平台播放命令
            if sys.platform == "win32":
                proc = subprocess.Popen(["start", file_path], shell=True)
            elif sys.platform == "darwin":
                proc = subprocess.Popen(["afplay", file_path])
            else:
                proc = subprocess.Popen(["aplay", file_path])

            # 记录进程用于后续关闭
            self._active_processes.add(proc.pid)
            proc.wait()

            # 播放完成后清理
            if os.path.exists(file_path):
                os.unlink(file_path)
            self._active_processes.discard(proc.pid)

        await asyncio.get_event_loop().run_in_executor(
            None,  # 使用默认执行器
            _play_and_monitor
        )

    async def stop_all_playback(self):
        """停止所有正在播放的音频"""
        if sys.platform == "win32":
            os.system("taskkill /f /im wmplayer.exe")
        else:
            for pid in self._active_processes.copy():
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
        self._active_processes.clear()

class VoiceLLM:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.chat = ChatTTS.Chat()
        # self.chat.load_models(compile=False, source='local', local_path='weight/pre_train')
        self.chat.load_models(compile=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.speaker = torch.load('weight/speaker/speaker_1_man.pth', map_location=self.device)
        self.audio_manager = AudioManager()
        if not hasattr(self.chat.pretrain_models['tokenizer'], 'pad_token'):
            self.chat.pretrain_models['tokenizer'].pad_token = "[PAD]"
            self.chat.pretrain_models['tokenizer'].pad_token_id = 0
        # wavs = self.chat.infer(text, params_refine_text=params_refine_text, params_infer_code=params_infer_code)
        # torchaudio.save("temp_output.wav", torch.from_numpy(wavs[0]), 24000)
        # os.system("start temp_output.wav")
    async def chat_tts(self, text: str, oral: int = 1, laugh: int = 0, bk: int = 3) -> np.ndarray:
        # 1. 准备参数（同步操作）
        params_infer_code = {
            'spk_emb': self.speaker,   # add sampled speaker 说话人嵌入，语音特征
            'temperature':.3,          # using custom temperature 控制生成文本的随机性
            'top_P': 0.7,              # top P decode 模型选择几个词使得概率之和等于top_P
            # 'prompt': '[speed_{}]'.format(speed)
        }

        params_refine_text = {
            'prompt': '[oral_{}][laugh_{}][break_{}]'.format(oral, laugh, bk)
        }

        # 2. 异步执行TTS生成
        wave_data = await asyncio.get_event_loop().run_in_executor(
            None,  # 使用默认执行器
            lambda: self.chat.infer(text, params_refine_text=params_refine_text, params_infer_code=params_infer_code)[0]
        )

        # 3. 异步播放并返回结果
        await self.audio_manager.play_temp_audio(wave_data)
        return wave_data

    async def ask_with_voice(self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        dialogue_examples: Optional[List[dict]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,) -> str:
        """
        带语音的LLM交互

        参数:
            messages: 消息列表
            system_msgs: 系统消息
            stream: 是否流式响应
            temperature: 温度参数

        返回:
            LLM的文本响应
        """
        self._is_speaking = True
        try:
            # 获取LLM响应
            response = await self.llm.voice_ask(
                    messages = messages,
                    system_msgs = system_msgs,
                    dialogue_examples = dialogue_examples,
                    stream = stream,
                    temperature = temperature
                )

            if stream:
                # 流式响应处理
                chunks = []
                async for chunk in response:
                    if chunk.strip():
                        chunks.append(chunk)
                        wavs = await self.chat_tts(text = response)
                return "".join(chunks)
            else:
                # 非流式响应处理
                if response.strip():
                    print(f"[LLM回复]: {response}")
                    wavs = await self.chat_tts(text = response)
                    # torchaudio.save("temp_output.wav", torch.from_numpy(wavs[0]), 24000)
                    # os.system("start temp_output.wav")
                return response

        except TokenLimitExceeded:
            logger.error("Token限制超出")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API错误: {oe}")
            raise
        except Exception as e:
            logger.error(f"语音交互异常: {e}")
            raise
        finally:
            self._is_speaking = False

    async def stop_playback(self):
        """立即停止当前播放"""
        await self.audio_manager.stop_all_playback()
