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
import cv2
import base64
from PIL import Image
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastConfig:
    # API Keys
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

    # Basic Config
    CITY_NAME = os.getenv("CITY_NAME", "Ahmedabad,IN")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    
    # Performance Settings
    DEEPGRAM_SAMPLE_RATE = 16000
    RECORDING_DURATION = 5  # seconds to record
    
    # Ultra-fast TTS settings
    ELEVENLABS_FAST_MODEL = "eleven_flash_v2_5"
    
    # Keywords for fast responses
    WEATHER_KEYWORDS = ["weather", "forecast", "temperature", "climate", "rain", "sunny"]
    TIME_KEYWORDS = ["time", "clock", "hour", "minute", "date", "day", "today"]
    
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
        logger.info("✅ All API keys validated")

# Validate config
FastConfig.validate()

# Initialize APIs
genai.configure(api_key=FastConfig.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ElevenLabs setup
try:
    from elevenlabs import ElevenLabs, VoiceSettings
    client = ElevenLabs(api_key=FastConfig.ELEVENLABS_API_KEY)
    logger.info("✅ ElevenLabs initialized")
except ImportError as e:
    logger.error(f"ElevenLabs error: {e}")
    exit(1)

class UltraFastAssistant:
    def __init__(self):
        self.cached_weather = None
        self.last_weather_update = 0
        self.weather_cache_duration = 300  # 5 minutes
        self.vision_keywords = ["see", "look", "what's there", "what is there", "describe", "show me", "can you see"]
        self.vision_model = None
        self._init_vision_model()
        
    def _init_vision_model(self):
        """Initialize the vision model if API key is available"""
        if FastConfig.GEMINI_API_KEY:
            try:
                genai.configure(api_key=FastConfig.GEMINI_API_KEY)
                # Use gemini-1.5-pro-latest which supports both text and vision
                self.vision_model = genai.GenerativeModel('gemini-1.5-pro-latest')
                logger.info("🔍 Vision model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize vision model: {e}")
                self.vision_model = None
        
        logger.info("🚀 Ultra-Fast Assistant Ready!")

    def record_audio_simple(self):
        """Simple audio recording - no streaming"""
        logger.info("🎙️ Recording... speak now!")
        
        try:
            # Record audio
            audio_data = sd.rec(
                int(FastConfig.RECORDING_DURATION * FastConfig.DEEPGRAM_SAMPLE_RATE),
                samplerate=FastConfig.DEEPGRAM_SAMPLE_RATE,
                channels=1,
                dtype=np.int16
            )
            
            # Show countdown
            for i in range(FastConfig.RECORDING_DURATION, 0, -1):
                print(f"Recording... {i}", end='\r')
                time.sleep(1)
            
            sd.wait()  # Wait for recording to finish
            print("Recording complete!   ")
            
            # Convert to WAV format
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
        """Simple HTTP-based transcription for maximum speed"""
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
            
            response = requests.post(
                url,
                headers=headers,
                params=params,
                data=audio_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check if any speech was detected
                if not result.get('results') or not result['results'].get('channels'):
                    logger.error("❌ No speech detected in the audio")
                    return ""
                    
                transcript = result['results']['channels'][0]['alternatives'][0].get('transcript', '')
                
                if not transcript.strip():
                    logger.error("❌ No speech detected in the audio")
                    return ""
                
                stt_time = time.time() - start_time
                logger.info(f"📝 STT ({stt_time:.2f}s): {transcript}")
                return transcript
            else:
                error_msg = response.text if response.text else "No error details provided"
                logger.error(f"Deepgram error {response.status_code}: {error_msg}")
                return ""
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def get_cached_weather(self):
        """Get weather with caching for speed"""
        current_time = time.time()
        
        if (self.cached_weather is None or 
            current_time - self.last_weather_update > self.weather_cache_duration):
            
            try:
                if not FastConfig.OPENWEATHER_API_KEY:
                    logger.error("OpenWeatherMap API key is not set")
                    return None
                    
                logger.info(f"🌤️ Fetching fresh weather data for {FastConfig.CITY_NAME}")
                url = f"http://api.openweathermap.org/data/2.5/weather?q={FastConfig.CITY_NAME}&appid={FastConfig.OPENWEATHER_API_KEY}&units=metric"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'main' in data and 'weather' in data and data['weather']:
                        self.cached_weather = {
                            "temp": data["main"]["temp"],
                            "condition": data["weather"][0]["description"],
                            "city": data.get("name", FastConfig.CITY_NAME.split(',')[0])
                        }
                        self.last_weather_update = current_time
                        logger.info(f"🌤️ Weather cached for {self.cached_weather['city']}")
                    else:
                        logger.error(f"Unexpected API response format: {data}")
                        return None
                else:
                    logger.error(f"Weather API error {response.status_code}: {response.text}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Weather API request failed: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error getting weather: {str(e)}")
                return None
                
        return self.cached_weather

    def capture_image(self):
        """Capture an image using the default camera"""
        try:
            # Initialize the camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Failed to open camera")
                return None
                
            # Capture a frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error("Failed to capture image")
                return None
                
            # Convert to RGB (OpenCV uses BGR by default)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Save the image for debugging
            timestamp = int(time.time())
            image_path = f"capture_{timestamp}.jpg"
            pil_image.save(image_path)
            logger.info(f"📸 Image captured and saved as {image_path}")
            
            return pil_image
            
        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            return None
            
    def analyze_image(self, image, prompt):
        """Analyze the captured image using Gemini"""
        if not self.vision_model:
            return "I'm sorry, the vision system is not available right now."
            
        try:
            # Prepare the prompt with more context
            full_prompt = (f"{prompt} Be concise and descriptive in your response. "
                         "If you can't see anything relevant, just say what you can see.")
            
            # Generate content with the image and prompt
            response = self.vision_model.generate_content([
                {
                    'parts': [
                        {'text': full_prompt},
                        {'inline_data': {
                            'mime_type': 'image/jpeg',
                            'data': self._pil_to_base64(image)
                        }}
                    ],
                    'role': 'user'
                }
            ])
            
            # Handle the response
            if response and hasattr(response, 'text'):
                return response.text
            else:
                logger.error(f"Unexpected response format: {response}")
                return "I'm not sure what I'm seeing. Could you try asking something else?"
                
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return "I had trouble analyzing the image. Please try again."
            
    def _pil_to_base64(self, image):
        """Convert PIL Image to base64"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def process_fast(self, question):
        """Lightning-fast question processing"""
        start_time = time.time()
        question_lower = question.lower()
        
        # Check if this is a vision-related question
        if any(keyword in question_lower for keyword in self.vision_keywords):
            logger.info("👀 Processing vision request...")
            # Capture image
            image = self.capture_image()
            if image:
                # Analyze the image
                response = self.analyze_image(image, question)
                return response
            return "I couldn't capture an image. Please check the camera."

        
        try:
            # Exit commands
            if any(word in question_lower for word in ["exit", "quit", "stop", "bye"]):
                return "EXIT"
            
            # Time queries
            elif any(word in question_lower for word in FastConfig.TIME_KEYWORDS):
                now = datetime.now()
                response = f"It's {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d')}"
            
            # Weather queries
            elif any(word in question_lower for word in FastConfig.WEATHER_KEYWORDS):
                if not FastConfig.OPENWEATHER_API_KEY:
                    response = "Weather service is not properly configured. Please check the OpenWeatherMap API key."
                else:
                    weather = self.get_cached_weather()
                    if weather:
                        response = f"It's {weather['temp']:.0f} degrees with {weather['condition']} in {weather['city']}"
                    else:
                        response = "I couldn't fetch the weather information. Please check your internet connection and API key."
            
            # General questions
            else:
                prompt = f"Answer in exactly one short sentence (maximum 15 words): {question}"
                response = gemini_model.generate_content(
                    prompt,
                    generation_config={
                        'max_output_tokens': 30,
                        'temperature': 0.1
                    }
                ).text.strip()
            
            process_time = time.time() - start_time
            logger.info(f"🧠 Processing ({process_time:.2f}s): {response}")
            return response
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return "Sorry, I had an error"

    def speak_fast(self, text):
        """Ultra-fast TTS with fallback to gTTS"""
        start_time = time.time()
        try:
            # First try ElevenLabs
            if self._try_elevenlabs_tts(text):
                return
                
            # If ElevenLabs fails, try gTTS
            if self._try_gtts(text):
                return
                
            # If both TTS methods fail, fall back to text
            logger.warning("All TTS methods failed, falling back to text")
            print(f"🔊 Assistant (text only): {text}")
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            print(f"🔊 Assistant (error): {text}")
        finally:
            tts_time = time.time() - start_time
            logger.info(f"🔊 TTS ({tts_time:.2f}s): {text}")
            
    def _try_elevenlabs_tts(self, text):
        """Try to use ElevenLabs TTS, return True if successful"""
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
                # Play the audio
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
        """Try to use gTTS, return True if successful"""
        try:
            logger.info("🔊 Trying gTTS fallback...")
            # Create gTTS object
            tts = gTTS(text=text, lang='en')
            
            # Save to bytes buffer
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Initialize pygame mixer if not already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            # Play the audio
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            return True
            
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            return False

    def run_continuous(self):
        """Run continuously without wake words"""
        logger.info("🎯 Continuous mode - Press ENTER to start recording, 'q' to quit")
        
        # Pre-cache weather
        self.get_cached_weather()
        
        while True:
            try:
                # Wait for user input
                user_input = input("\nPress ENTER to start recording (or 'q' to quit): ").strip().lower()
                
                if user_input in ['q', 'quit', 'exit']:
                    logger.info("👋 Goodbye!")
                    break
                
                # Record and process
                total_start = time.time()
                
                # Step 1: Record
                audio_data = self.record_audio_simple()
                if not audio_data:
                    continue
                
                # Step 2: Transcribe
                text = self.transcribe_simple(audio_data)
                if not text or not text.strip():
                    print("❌ No speech detected")
                    continue
                
                # Step 3: Process
                response = self.process_fast(text)
                if response == "EXIT":
                    logger.info("👋 Goodbye!")
                    break
                
                # Step 4: Speak
                self.speak_fast(response)
                
                # Performance summary
                total_time = time.time() - total_start
                logger.info(f"⚡ Total response time: {total_time:.2f}s")
                
            except KeyboardInterrupt:
                logger.info("👋 Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(0.5)

def main():
    """Main entry point"""
    assistant = UltraFastAssistant()
    
    print("""
🚀 ULTRA-FAST VOICE ASSISTANT 
===============================
- No wake words needed
- Press ENTER to start recording
- Speak for 5 seconds when prompted
- Type 'q' to quit
    """)
    
    assistant.run_continuous()

if __name__ == "__main__":
    main()
