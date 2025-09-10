#!/usr/bin/env python
import asyncio
import os
import json
import requests
import sounddevice as sd
import numpy as np
import google.generativeai as genai
import time
import logging
from dotenv import load_dotenv
from datetime import datetime
import wave
import io
from gtts import gTTS
import pygame
from io import BytesIO
import RPi.GPIO as GPIO
import cv2

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastConfig:
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    CITY_NAME = os.getenv("CITY_NAME", "Ahmedabad,IN")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    DEEPGRAM_SAMPLE_RATE = 16000
    RECORDING_DURATION = 5
    ELEVENLABS_FAST_MODEL = "eleven_flash_v2_5"
    WEATHER_KEYWORDS = ["weather", "forecast", "temperature", "climate", "rain", "sunny"]
    TIME_KEYWORDS = ["time", "clock", "hour", "minute", "date", "day", "today"]
    CV_KEYWORDS = ["see", "what is this", "describe", "identify", "recognize"]
    REPEAT_KEYWORDS = ["repeat", "say that again", "can you repeat", "once more"]

    @classmethod
    def validate(cls):
        required_keys = {
            "ELEVENLABS_API_KEY": cls.ELEVENLABS_API_KEY,
            "DEEPGRAM_API_KEY": cls.DEEPGRAM_API_KEY,
            "GEMINI_API_KEY": cls.GEMINI_API_KEY
        }
        missing_keys = [name for name, key in required_keys.items() if not key]
        if missing_keys:
            logger.error("Missing API keys!")
            for key in missing_keys:
                logger.error(f"   {key}=your_key_here")
            raise ValueError(f"Missing keys: {', '.join(missing_keys)}")
        logger.info("✔ All API keys validated")

FastConfig.validate()
genai.configure(api_key=FastConfig.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

try:
    from elevenlabs import ElevenLabs, VoiceSettings
    client = ElevenLabs(api_key=FastConfig.ELEVENLABS_API_KEY)
    logger.info("✔ ElevenLabs initialized")
except ImportError as e:
    logger.error(f"ElevenLabs error: {e}")
    exit(1)

class LCDDisplay:
    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        self.LCD_RS = 25
        self.LCD_E = 24
        self.LCD_D4 = 23
        self.LCD_D5 = 17
        self.LCD_D6 = 18
        self.LCD_D7 = 22
        self.LCD_WIDTH = 16
        self.LCD_CHR = True
        self.LCD_CMD = False
        self.LCD_LINE_1 = 0x80
        self.LCD_LINE_2 = 0xC0
        self.E_PULSE = 0.0005
        self.E_DELAY = 0.0005
        GPIO.setup(self.LCD_E, GPIO.OUT)
        GPIO.setup(self.LCD_RS, GPIO.OUT)
        GPIO.setup(self.LCD_D4, GPIO.OUT)
        GPIO.setup(self.LCD_D5, GPIO.OUT)
        GPIO.setup(self.LCD_D6, GPIO.OUT)
        GPIO.setup(self.LCD_D7, GPIO.OUT)
        self.lcd_init()

    def lcd_init(self):
        self.lcd_display(0x28, self.LCD_CMD)
        self.lcd_display(0x0C, self.LCD_CMD)
        self.lcd_display(0x01, self.LCD_CMD)
        time.sleep(self.E_DELAY)

    def lcd_display(self, bits, mode):
        GPIO.output(self.LCD_RS, mode)
        GPIO.output(self.LCD_D4, False)
        GPIO.output(self.LCD_D5, False)
        GPIO.output(self.LCD_D6, False)
        GPIO.output(self.LCD_D7, False)
        if bits & 0x10 == 0x10:
            GPIO.output(self.LCD_D4, True)
        if bits & 0x20 == 0x20:
            GPIO.output(self.LCD_D5, True)
        if bits & 0x40 == 0x40:
            GPIO.output(self.LCD_D6, True)
        if bits & 0x80 == 0x80:
            GPIO.output(self.LCD_D7, True)
        self.lcd_toggle_enable()
        GPIO.output(self.LCD_D4, False)
        GPIO.output(self.LCD_D5, False)
        GPIO.output(self.LCD_D6, False)
        GPIO.output(self.LCD_D7, False)
        if bits & 0x01 == 0x01:
            GPIO.output(self.LCD_D4, True)
        if bits & 0x02 == 0x02:
            GPIO.output(self.LCD_D5, True)
        if bits & 0x04 == 0x04:
            GPIO.output(self.LCD_D6, True)
        if bits & 0x08 == 0x08:
            GPIO.output(self.LCD_D7, True)
        self.lcd_toggle_enable()

    def lcd_toggle_enable(self):
        time.sleep(self.E_DELAY)
        GPIO.output(self.LCD_E, True)
        time.sleep(self.E_PULSE)
        GPIO.output(self.LCD_E, False)
        time.sleep(self.E_DELAY)

    def lcd_string(self, message, line):
        message = message.ljust(self.LCD_WIDTH, " ")
        self.lcd_display(line, self.LCD_CMD)
        for i in range(self.LCD_WIDTH):
            self.lcd_display(ord(message[i]), self.LCD_CHR)

    def clear(self):
        self.lcd_display(0x01, self.LCD_CMD)

    def cleanup(self):
        self.clear()
        GPIO.cleanup()

class UltraFastAssistant:
    def __init__(self):
        self.cached_weather = None
        self.last_weather_update = 0
        self.weather_cache_duration = 300
        self.lcd = LCDDisplay()
        self.last_response = ""
        logger.info("🟢 Ultra-Fast Assistant Ready!")

    def record_audio_simple(self):
        logger.info("🎙 Recording... speak now!")
        try:
            audio_data = sd.rec(
                int(FastConfig.RECORDING_DURATION * FastConfig.DEEPGRAM_SAMPLE_RATE),
                samplerate=FastConfig.DEEPGRAM_SAMPLE_RATE,
                channels=1,
                dtype=np.int16
            )
            for i in range(FastConfig.RECORDING_DURATION, 0, -1):
                print(f"Recording... {i}", end='\r')
                time.sleep(1)
            sd.wait()
            print("Recording complete!   ")
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(FastConfig.DEEPGRAM_SAMPLE_RATE)
                wav_file.writeframes(audio_data.tobytes())
            return wav_buffer.getvalue()
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None

    def transcribe_simple(self, audio_data):
        start_time = time.time()
        try:
            url = "https://api.deepgram.com/v1/listen"
            headers = {
                "Authorization": f"Token {FastConfig.DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav"
            }
            params = {
                "model": "nova-2-general",
                "punctuate": "true",
                "smart_format": "true"
            }
            response = requests.post(url, headers=headers, params=params, data=audio_data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if not result.get('results') or not result['results'].get('channels'):
                    logger.error("❌ No speech detected in the audio")
                    return ""
                transcript = result['results']['channels'][0]['alternatives'][0].get('transcript', '')
                if not transcript.strip():
                    logger.error("❌ No speech detected in the audio")
                    return ""
                stt_time = time.time() - start_time
                logger.info(f"📥 STT ({stt_time:.2f}s): {transcript}")
                return transcript
            else:
                error_msg = response.text if response.text else "No error details provided"
                logger.error(f"Deepgram error {response.status_code}: {error_msg}")
                return ""
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def get_cached_weather(self):
        current_time = time.time()
        if self.cached_weather is None or current_time - self.last_weather_update > self.weather_cache_duration:
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather?q={FastConfig.CITY_NAME}&appid={FastConfig.OPENWEATHER_API_KEY}&units=metric"
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    self.cached_weather = {
                        "temp": data["main"]["temp"],
                        "condition": data["weather"][0]["description"],
                        "city": data["name"]
                    }
                    self.last_weather_update = current_time
                    logger.info("🌤 Weather cached")
            except Exception as e:
                logger.error(f"Weather error: {e}")
        return self.cached_weather

    def handle_cv_request(self):
        try:
            self.lcd.lcd_string("Capturing...", self.lcd.LCD_LINE_1)
            self.lcd.lcd_string("Please wait", self.lcd.LCD_LINE_2)
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                logger.error("❌ Camera capture failed")
                return "Sorry, I couldn't capture the image"
            image_path = "/tmp/captured.jpg"
            cv2.imwrite(image_path, frame)
            description = "It looks like a cup"  # Placeholder, integrate real CV here
            logger.info("✅ Image processed")
            return description
        except Exception as e:
            logger.error(f"CV error: {e}")
            return "Sorry, I couldn't process the image"

    def process_fast(self, question):
        start_time = time.time()
        question_lower = question.lower()
        try:
            if any(word in question_lower for word in ["exit", "quit", "stop", "bye"]):
                return "EXIT"
            elif any(word in question_lower for word in FastConfig.REPEAT_KEYWORDS):
                if self.last_response:
                    return self.last_response
                else:
                    return "Nothing to repeat yet"
            elif any(word in question_lower for word in FastConfig.TIME_KEYWORDS):
                now = datetime.now()
                response = f"It's {now.strftime('%I:%M %p')} on {now.strftime('%A')}"
                self.lcd.lcd_string("Current Time:", self.lcd.LCD_LINE_1)
                self.lcd.lcd_string(response, self.lcd.LCD_LINE_2)
            elif any(word in question_lower for word in FastConfig.WEATHER_KEYWORDS):
                weather = self.get_cached_weather()
                if weather:
                    response = f"{weather['temp']:.0f}°C, {weather['condition']}"
                    self.lcd.lcd_string("Weather in:", self.lcd.LCD_LINE_1)
                    self.lcd.lcd_string(weather['city'], self.lcd.LCD_LINE_2)
                else:
                    response = "Weather info unavailable"
                    self.lcd.lcd_string("Weather:", self.lcd.LCD_LINE_1)
                    self.lcd.lcd_string("Unavailable", self.lcd.LCD_LINE_2)
            elif any(word in question_lower for word in FastConfig.CV_KEYWORDS):
                response = self.handle_cv_request()
            else:
                prompt = f"Answer in exactly one short sentence (maximum 15 words): {question}"
                response = gemini_model.generate_content(
                    prompt,
                    generation_config={
                        'max_output_tokens': 30,
                        'temperature': 0.1
                    }
                ).text.strip()
            self.last_response = response
            process_time = time.time() - start_time
            logger.info(f"⏱ Processing ({process_time:.2f}s): {response}")
            return response
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return "Sorry, I had an error"

    def speak_fast(self, text):
        start_time = time.time()
        try:
            if self._try_elevenlabs_tts(text):
                return
            if self._try_gtts(text):
                return
            logger.warning("All TTS methods failed, falling back to text")
            print(f"💬 Assistant (text only): {text}")
        except Exception as e:
            logger.error(f"TTS error: {e}")
            print(f"💬 Assistant (error): {text}")
        finally:
            tts_time = time.time() - start_time
            logger.info(f"🔊 TTS ({tts_time:.2f}s): {text}")

    def _try_elevenlabs_tts(self, text):
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{FastConfig.ELEVENLABS_VOICE_ID}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": FastConfig.ELEVENLABS_API_KEY
            }
            data = {
                "text": text,
                "model_id": FastConfig.ELEVENLABS_FAST_MODEL,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            response = requests.post(url, json=data, headers=headers, timeout=10)
            if response.status_code == 200:
                audio_data = BytesIO(response.content)
                pygame.mixer.init()
                pygame.mixer.music.load(audio_data)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                return True
            else:
                logger.warning(f"ElevenLabs TTS failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.warning(f"ElevenLabs TTS error: {e}")
            return False

    def _try_gtts(self, text):
        try:
            logger.info("🔊 Trying gTTS fallback...")
            tts = gTTS(text=text, lang='en')
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            return True
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            return False

    def run_continuous(self):
        logger.info("🎛 Continuous mode - Press ENTER to start recording, 'q' to quit")
        self.get_cached_weather()
        try:
            while True:
                user_input = input("\nPress ENTER to start recording (or 'q' to quit): ").strip().lower()
                if user_input in ['q', 'quit', 'exit']:
                    logger.info("👋 Goodbye!")
                    break
                total_start = time.time()
                audio_data = self.record_audio_simple()
                if not audio_data:
                    continue
                text = self.transcribe_simple(audio_data)
                if not text or not text.strip():
                    print("❌ No speech detected")
                    continue
                response = self.process_fast(text)
                if response == "EXIT":
                    logger.info("👋 Goodbye!")
                    break
                self.speak_fast(response)
                total_time = time.time() - total_start
                logger.info(f"⚙ Total response time: {total_time:.2f}s")
        except KeyboardInterrupt:
            logger.info("👋 Interrupted by user")
        finally:
            self.lcd.cleanup()

def main():
    assistant = UltraFastAssistant()
    print("""
🟢 ULTRA-FAST VOICE ASSISTANT
===============================
- No wake words needed
- Press ENTER to start recording
- Speak for 5 seconds when prompted
- Type 'q' to quit
""")
    assistant.run_continuous()

if __name__ == "__main__":
    main()
