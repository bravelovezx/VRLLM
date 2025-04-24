import asyncio
import numpy as np
import sounddevice as sd
from torch import device
import whisper
from typing import AsyncGenerator, Optional
from dataclasses import dataclass
from app.logger import logger

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    silence_threshold: float = 0.02
    min_voice_duration: float = 0.5

class SpeachRecognizer:
    def __init__(
        self,
        wake_word: str,  # 必须传入唤醒词
        model_size: str = "base",
        device_index: Optional[int] = None,
    ):
        """
        严格唤醒词识别器（检测到唤醒词后才开始正式识别）

        参数:
            wake_word: 必须的唤醒词（如"小助手"）
            model_size: whisper模型大小
            device_index: 音频输入设备索引
        """
        self.model = whisper.load_model(model_size)
        self.config = AudioConfig()
        self.device_index = device_index if device_index is not None else sd.default.device[0]
        self.wake_word = wake_word.lower()
        self._is_listening = False
        self.device()  # 检查设备兼容性
        logger.info(f"🎧 使用音频设备: {self.device_index}")
        logger.info(f"🤖 语音识别器初始化完成，唤醒词: '{self.wake_word}'")
        

    async def _record_audio_chunk(self, duration: float) -> np.ndarray:
        """录制音频片段"""
        recording = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: sd.rec(
                int(duration * self.config.sample_rate),
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                device=self.device_index,
                dtype=self.config.dtype,
                blocking=True,
            ),
        )
        return recording.squeeze()

    def device(self):
        try:
            sd.check_input_settings(
                device=self.device_index,
                samplerate=self.config.sample_rate,
                channels=self.config.channels
            )
        except sd.PortAudioError as e:
            logger.error(f"Device incompatibility: {str(e)}")
            logger.info("Please select from the following valid input devices：")
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] > 0:
                    logger.info(f"Index {i}: {dev['name']}")
            exit(1)

    async def _vad_process(self, audio_data: np.ndarray) -> bool:
        """语音活动检测（跳过静音片段）"""
        energy = np.mean(np.abs(audio_data))
        result = energy > self.config.silence_threshold
        logger.debug(f"🔊 语音活动检测: {'有语音' if result else '无语音'}")
        return result

    async def _check_wake_word(self, audio_data: np.ndarray) -> bool:
        """严格检测唤醒词（专为唤醒优化）"""
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # 唤醒词检测专用参数（提高响应速度）
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio_data,
                language="zh",
                no_speech_threshold=0.6,           # 更高阈值避免误唤醒
                temperature=0.0,                   # 禁用随机性
                condition_on_previous_text=False,  # 不依赖上文
            ),
        )
        detected = self.wake_word in result["text"].strip().lower()
        logger.debug(f"👂 唤醒词检测: {'检测到' if detected else '未检测到'}唤醒词 '{self.wake_word}'")
        return detected

    async def wait_for_wake_word(self) -> None:
        """阻塞直到检测到唤醒词"""
        self._is_listening = True
        logger.info(f"🔇 休眠中，等待唤醒词: '{self.wake_word}'...")

        while self._is_listening:
            audio_data = await self._record_audio_chunk(1.0)  # 1秒片段提高唤醒词检测率

            if await self._vad_process(audio_data) and await self._check_wake_word(audio_data):
                logger.info(f"✅ 检测到唤醒词！开始监听...")
                return

    async def continuous_recognition(self) -> AsyncGenerator[str, None]:
        """正式语音识别（必须在检测到唤醒词后调用）"""
        try:
            while self._is_listening:
                audio_data = await self._record_audio_chunk(5)

                if not await self._vad_process(audio_data):
                    continue

                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.transcribe(
                        audio_data,
                        language="zh",
                        no_speech_threshold=0.3,
                        temperature=0.2,
                    ),
                )

                text = result["text"].strip()
                if text:
                    logger.info(f"🗣️ 识别结果: {text}")
                    yield text

        finally:
            self.stop()

    def stop(self):
        """停止识别"""
        self._is_listening = False
        logger.info("🛑 识别已停止")
