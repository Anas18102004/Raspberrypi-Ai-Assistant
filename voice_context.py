#!/usr/bin/env python
import os
import json
import time
import wave
import io
import logging
import requests
import threading
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import sounddevice as sd
from gtts import gTTS
import pygame
import cv2
import RPi.GPIO as GPIO
import google.generativeai as genai
from io import BytesIO

# -------------------- CONFIGURATION --------------------
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
    WEATHER_CACHE_DURATION = 300
    TIME_KEYWORDS = {"time", "clock", "hour", "minute", "date", "day", "today"}
    WEATHER_KEYWORDS = {"weather", "forecast", "temperature", "climate", "rain", "sunny"}
    CV_KEYWORDS = {"see", "what is this", "describe", "identify", "recognize"}
    REPEAT_KEYWORDS = {"repeat", "say that again", "can you repeat", "once more"}

    @classmethod
    def validate(cls):
        required = {
            "ELEVENLABS_API_KEY": cls.ELEVENLABS_API_KEY,
            "OPENWEATHER_API_KEY": cls.OPENWEATHER_API_KEY,
            "GEMINI_API_KEY": cls.GEMINI_API_KEY,
            "DEEPGRAM_API_KEY": cls.DEEPGRAM_API_KEY
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            logger.error(f"Missing environment variables: {', '.join(missing)}")
            raise ValueError("Check your .env file.")
        logger.info("✔ Environment validated.")

FastConfig.validate()

# -------------------- GENERATIVE MODEL --------------------
genai.configure(api_key=FastConfig.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# -------------------- SESSION --------------------
session = requests.Session()

# -------------------- LCD DISPLAY --------------------
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
        GPIO.output(self.LCD_D4, bits & 0x10 == 0x10)
        GPIO.output(self.LCD_D5, bits & 0x20 == 0x20)
        GPIO.output(self.LCD_D6, bits & 0x40 == 0x40)
        GPIO.output(self.LCD_D7, bits & 0x80 == 0x80)
        self.lcd_toggle_enable()
        GPIO.output(self.LCD_D4, bits & 0x01 == 0x01)
        GPIO.output(self.LCD_D5, bits & 0x02 == 0x02)
        GPIO.output(self.LCD_D6, bits & 0x04 == 0x04)
        GPIO.output(self.LCD_D7, bits & 0x08 == 0x08)
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
        for char in message:
            self.lcd_display(ord(char), self.LCD_CHR)

    def clear(self):
        self.lcd_display(0x01, self.LCD_CMD)

    def cleanup(self):
        self.clear()
        GPIO.cleanup()

# -------------------- ULTRA FAST ASSISTANT --------------------
class UltraFastAssistant:
    def __init__(self):
        self.weather_cache = None
        self.weather_last_update = 0
        self.lcd = LCDDisplay()
        self.last_response = ""
        self.cached_answers = {}
        self.session = session
        self.pygame_initialized = False
        self.weather_lock = threading.Lock()
        logger.info("✅ Assistant initialized")

    def record_audio(self):
        logger.info("🎙 Recording...")
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

    def transcribe_audio(self, audio_data):
        logger.info("⏳ Transcribing...")
        try:
            url = "https://api.deepgram.com/v1/listen"
            headers = {"Authorization": f"Token {FastConfig.DEEPGRAM_API_KEY}"}
            params = {"model": "nova-2-general", "punctuate": "true", "smart_format": "true"}
            response = self.session.post(url, headers=headers, params=params, data=audio_data, timeout=5)
            result = response.json()
            transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            logger.info(f"✅ Transcription: {transcript}")
            return transcript
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return ""

    def get_weather(self):
        with self.weather_lock:
            now = time.time()
            if self.weather_cache and now - self.weather_last_update < FastConfig.WEATHER_CACHE_DURATION:
                return self.weather_cache
            try:
                url = f"http://api.openweathermap.org/data/2.5/weather"
                params = {
                    "q": FastConfig.CITY_NAME,
                    "appid": FastConfig.OPENWEATHER_API_KEY,
                    "units": "metric"
                }
                response = self.session.get(url, params=params, timeout=3)
                data = response.json()
                weather = {
                    "temp": data["main"]["temp"],
                    "desc": data["weather"][0]["description"],
                    "city": data["name"]
                }
                self.weather_cache = weather
                self.weather_last_update = now
                logger.info("✅ Weather updated")
                return weather
            except Exception as e:
                logger.error(f"❌ Weather fetch failed: {e}")
                return None

    def handle_cv_request(self):
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
        logger.info("✅ Image captured")
        return "It looks like a cup"

    def speak(self, text):
        logger.info("🔊 Speaking...")
        self.last_response = text
        try:
            if not self.pygame_initialized:
                pygame.mixer.init()
                self.pygame_initialized = True
            # Try ElevenLabs first
            if self._try_elevenlabs(text):
                return
            # Fallback to gTTS
            self._try_gtts(text)
        except Exception as e:
            logger.error(f"❌ TTS failed: {e}")

    def _try_elevenlabs(self, text):
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
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
            }
            response = self.session.post(url, json=data, headers=headers, timeout=5)
            if response.status_code == 200:
                audio_data = BytesIO(response.content)
                pygame.mixer.music.load(audio_data)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                return True
            else:
                logger.warning(f"ElevenLabs error: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"ElevenLabs failed: {e}")
            return False

    def _try_gtts(self, text):
        try:
            tts = gTTS(text=text, lang="en")
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            return True
        except Exception as e:
            logger.error(f"gTTS failed: {e}")
            return False

    def process(self, question):
        logger.info(f"Processing: {question}")
        q = question.lower()
        if q in self.cached_answers:
            return self.cached_answers[q]
        if any(word in q for word in FastConfig.REPEAT_KEYWORDS):
            return self.last_response or "Nothing to repeat"
        if any(word in q for word in FastConfig.TIME_KEYWORDS):
            now = datetime.now()
            result = f"It's {now.strftime('%I:%M %p')} on {now.strftime('%A')}"
            self.lcd.lcd_string("Current Time:", self.lcd.LCD_LINE_1)
            self.lcd.lcd_string(result, self.lcd.LCD_LINE_2)
            self.cached_answers[q] = result
            return result
        if any(word in q for word in FastConfig.WEATHER_KEYWORDS):
            weather = self.get_weather()
            if weather:
                result = f"{weather['temp']:.0f}°C, {weather['desc']}"
                self.lcd.lcd_string("Weather Info:", self.lcd.LCD_LINE_1)
                self.lcd.lcd_string(weather["city"], self.lcd.LCD_LINE_2)
                self.cached_answers[q] = result
                return result
            else:
                result = "Weather unavailable"
                self.lcd.lcd_string("Weather:", self.lcd.LCD_LINE_1)
                self.lcd.lcd_string("Unavailable", self.lcd.LCD_LINE_2)
                return result
        if any(word in q for word in FastConfig.CV_KEYWORDS):
            result = self.handle_cv_request()
            self.cached_answers[q] = result
            return result
        # Default: Use Gemini model
        try:
            prompt = f"Answer briefly in one sentence: {question}"
            response = gemini_model.generate_content(
                prompt,
                generation_config={"max_output_tokens": 30, "temperature": 0.1}
            ).text.strip()
            self.cached_answers[q] = response
            return response
        except Exception as e:
            logger.error(f"❌ Gemini failed: {e}")
            return "Sorry, I couldn't process that"

    def run(self):
        logger.info("🟢 Assistant running. Press ENTER to talk, 'q' to quit.")
        self.get_weather()
        try:
            while True:
                user_input = input("\nPress ENTER to record (or 'q' to quit): ").strip().lower()
                if user_input in ['q', 'quit', 'exit']:
                    logger.info("👋 Exiting.")
                    break
                threading.Thread(target=self.interaction).start()
        except KeyboardInterrupt:
            logger.info("👋 Interrupted by user.")
        finally:
            self.lcd.cleanup()

    def interaction(self):
        audio_data = self.record_audio()
        question = self.transcribe_audio(audio_data)
        if not question:
            logger.warning("❌ No speech detected")
            return
        response = self.process(question)
        logger.info(f"Response: {response}")
        self.speak(response)

# -------------------- MAIN --------------------
def main():
    assistant = UltraFastAssistant()
    print("""
🟢 ULTRA FAST VOICE ASSISTANT
=============================
- Press ENTER to start recording
- Say commands like:
  - "What's the weather?"
  - "What's the time?"
  - "Can you see this?" (CV mode)
  - "Repeat it please"
- Type 'q' to quit
""")
    assistant.run()

if __name__ == "__main__":
    main()
