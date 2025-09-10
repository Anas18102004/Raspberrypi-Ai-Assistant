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
import threading
from queue import Queue
import sys

# GPIO and LCD imports
try:
    import RPi.GPIO as GPIO
    from RPLCD.gpio import CharLCD
    GPIO_AVAILABLE = True
    print("✅ GPIO libraries loaded successfully")
except ImportError as e:
    print(f"⚠️ GPIO libraries not available: {e}")
    print("📝 Running in simulation mode (no actual LCD control)")
    GPIO_AVAILABLE = False

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
    SILENCE_THRESHOLD = 0.01  # Threshold for silence detection
    MIN_RECORDING_DURATION = 1.0  # Minimum recording time
    
    # Ultra-fast TTS settings
    ELEVENLABS_FAST_MODEL = "eleven_flash_v2_5"
    
    # LCD GPIO Configuration (16x2 LCD)
    LCD_CONFIG = {
        'rs': 25,        # GPIO 25 (Pin 22) - R5 connection
        'enable': 24,    # GPIO 24 (Pin 18) - E connection  
        'data_pins': [23, 17, 18, 22],  # GPIO 23,17,18,22 (Pins 16,11,12,15) - D4,D5,D6,D7
        'numbering_mode': GPIO.BCM,
        'cols': 16,
        'rows': 2,
        'dotsize': 8,
        'auto_linebreaks': True
    }
    
    # LED Configuration for status indication
    LED_CONFIG = {
        'blue_led': 21,   # GPIO 21 (Pin 40) - Blue LED
        'white_led': 20   # GPIO 20 (Pin 38) - White LED  
    }
    
    # Keywords for fast responses
    WEATHER_KEYWORDS = ["weather", "forecast", "temperature", "climate", "rain", "sunny", "cold", "hot"]
    TIME_KEYWORDS = ["time", "clock", "hour", "minute", "date", "day", "today", "now"]
    GREETING_KEYWORDS = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    LCD_KEYWORDS = ["display", "show", "screen", "lcd"]
    
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

class LCDController:
    """Handles LCD display operations"""
    
    def __init__(self):
        self.lcd = None
        self.is_initialized = False
        
        if GPIO_AVAILABLE:
            try:
                # Setup GPIO
                GPIO.setmode(FastConfig.LCD_CONFIG['numbering_mode'])
                GPIO.setwarnings(False)
                
                # Initialize LCD
                self.lcd = CharLCD(
                    pin_rs=FastConfig.LCD_CONFIG['rs'],
                    pin_e=FastConfig.LCD_CONFIG['enable'],
                    pins_data=FastConfig.LCD_CONFIG['data_pins'],
                    numbering_mode=FastConfig.LCD_CONFIG['numbering_mode'],
                    cols=FastConfig.LCD_CONFIG['cols'],
                    rows=FastConfig.LCD_CONFIG['rows'],
                    dotsize=FastConfig.LCD_CONFIG['dotsize'],
                    auto_linebreaks=FastConfig.LCD_CONFIG['auto_linebreaks']
                )
                
                # Setup status LEDs
                for led_name, pin in FastConfig.LED_CONFIG.items():
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, GPIO.LOW)
                
                self.is_initialized = True
                self.show_startup_message()
                logger.info("✅ LCD Controller initialized")
                
            except Exception as e:
                logger.error(f"LCD initialization error: {e}")
                self.is_initialized = False
        else:
            logger.info("📺 LCD Controller running in simulation mode")
    
    def show_startup_message(self):
        """Display startup message"""
        if self.is_initialized:
            self.lcd.clear()
            self.lcd.write_string("Voice Assistant\nStarting up...")
            time.sleep(2)
            self.clear_display()
        else:
            print("📺 [LCD SIMULATION] Voice Assistant\n📺 [LCD SIMULATION] Starting up...")
    
    def clear_display(self):
        """Clear the LCD display"""
        if self.is_initialized:
            self.lcd.clear()
        else:
            print("📺 [LCD SIMULATION] Display cleared")
    
    def display_text(self, line1, line2=""):
        """Display text on LCD"""
        if self.is_initialized:
            self.lcd.clear()
            # Truncate text to fit display
            line1 = line1[:16] if len(line1) > 16 else line1
            line2 = line2[:16] if len(line2) > 16 else line2
            
            self.lcd.write_string(line1)
            if line2:
                self.lcd.cursor_pos = (1, 0)
                self.lcd.write_string(line2)
        else:
            print(f"📺 [LCD SIMULATION] Line 1: {line1}")
            if line2:
                print(f"📺 [LCD SIMULATION] Line 2: {line2}")
    
    def display_weather(self, weather_data):
        """Display weather information"""
        if weather_data:
            temp = f"{weather_data['temp']:.0f}C"
            condition = weather_data['condition'][:12]  # Truncate condition
            
            self.display_text(f"{weather_data['city'][:10]} {temp}", condition.title())
        else:
            self.display_text("Weather", "Unavailable")
    
    def display_time(self):
        """Display current time and date"""
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%d/%m/%Y")
        
        self.display_text(time_str, date_str)
    
    def display_status(self, status, details=""):
        """Display status messages"""
        self.display_text(status[:16], details[:16])
    
    def set_led_status(self, led_name, state):
        """Control status LEDs"""
        if self.is_initialized and led_name in FastConfig.LED_CONFIG:
            pin = FastConfig.LED_CONFIG[led_name]
            GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
        else:
            state_text = "ON" if state else "OFF"
            print(f"💡 [LED SIMULATION] {led_name.upper()}: {state_text}")
    
    def indicate_listening(self):
        """Visual indication when listening"""
        self.set_led_status('blue_led', True)
        self.display_status("Listening...", "Speak now")
    
    def indicate_processing(self):
        """Visual indication when processing"""
        self.set_led_status('blue_led', False)
        self.set_led_status('white_led', True)
        self.display_status("Processing...", "Please wait")
    
    def indicate_speaking(self):
        """Visual indication when speaking"""
        self.set_led_status('white_led', False)
        self.set_led_status('blue_led', True)
        self.display_status("Speaking...", "")
    
    def indicate_ready(self):
        """Visual indication when ready"""
        self.set_led_status('blue_led', False)
        self.set_led_status('white_led', False)
        self.display_status("Ready", "Press ENTER")
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if self.is_initialized:
            try:
                self.clear_display()
                for led_name in FastConfig.LED_CONFIG:
                    self.set_led_status(led_name, False)
                GPIO.cleanup()
                logger.info("✅ LCD Controller cleaned up")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

class UltraFastAssistant:
    def __init__(self):
        self.cached_weather = None
        self.last_weather_update = 0
        self.weather_cache_duration = 300  # 5 minutes
        
        # Initialize LCD controller
        self.lcd = LCDController()
        
        # Conversation context for better responses
        self.conversation_context = []
        self.max_context_length = 3
        
        # Audio processing optimization
        self.audio_queue = Queue()
        self.is_processing = False
        
        # Initialize pygame mixer
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
        
        logger.info("🚀 Ultra-Fast Assistant with LCD Ready!")

    def detect_silence(self, audio_data, threshold=None):
        """Detect if audio contains mostly silence"""
        if threshold is None:
            threshold = FastConfig.SILENCE_THRESHOLD
            
        # Calculate RMS (Root Mean Square) of the audio
        rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        return rms < threshold

    def record_audio_adaptive(self):
        """Adaptive audio recording with LCD feedback"""
        logger.info("🎙️ Recording... speak now!")
        self.lcd.indicate_listening()
        
        try:
            # Start recording
            recorded_chunks = []
            chunk_duration = 0.5  # 500ms chunks
            chunk_samples = int(chunk_duration * FastConfig.DEEPGRAM_SAMPLE_RATE)
            
            start_time = time.time()
            speech_detected = False
            silence_count = 0
            max_silence_chunks = 3  # Stop after 1.5s of silence following speech
            
            for i in range(int(FastConfig.RECORDING_DURATION / chunk_duration)):
                # Record a chunk
                chunk = sd.rec(chunk_samples, 
                             samplerate=FastConfig.DEEPGRAM_SAMPLE_RATE,
                             channels=1, dtype=np.int16)
                sd.wait()
                
                recorded_chunks.append(chunk)
                
                # Check for speech
                if not self.detect_silence(chunk):
                    speech_detected = True
                    silence_count = 0
                    self.lcd.display_status("Recording...", f"{len(recorded_chunks) * chunk_duration:.1f}s")
                    print(f"🎤 Speaking... {len(recorded_chunks) * chunk_duration:.1f}s", end='\r')
                else:
                    if speech_detected:
                        silence_count += 1
                        self.lcd.display_status("Finishing...", f"Silence {silence_count}")
                        print(f"⏸️  Silence... {silence_count}", end='\r')
                        
                        # Stop early if we've detected speech and then silence
                        if silence_count >= max_silence_chunks:
                            elapsed = time.time() - start_time
                            if elapsed >= FastConfig.MIN_RECORDING_DURATION:
                                break
                    else:
                        self.lcd.display_status("Waiting...", f"{len(recorded_chunks) * chunk_duration:.1f}s")
                        print(f"🔇 Waiting for speech... {len(recorded_chunks) * chunk_duration:.1f}s", end='\r')
            
            print("\nRecording complete!   ")
            
            # Combine all chunks
            if recorded_chunks:
                audio_data = np.concatenate(recorded_chunks, axis=0)
                
                # Convert to WAV format
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(FastConfig.DEEPGRAM_SAMPLE_RATE)
                    wav_file.writeframes(audio_data.tobytes())
                
                return wav_buffer.getvalue()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Recording error: {e}")
            self.lcd.display_status("Record Error", "Try again")
            return None

    def transcribe_simple(self, audio_data):
        """Simple HTTP-based transcription for maximum speed"""
        start_time = time.time()
        self.lcd.indicate_processing()
        
        try:
            url = "https://api.deepgram.com/v1/listen"
            headers = {
                "Authorization": f"Token {FastConfig.DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav"
            }
            
            params = {
                "model": "nova-2-general",
                "punctuate": "true",
                "smart_format": "true",
                "diarize": "false",  # Disable diarization for speed
                "utterances": "false"  # Disable utterance detection for speed
            }
            
            self.lcd.display_status("Transcribing", "Please wait...")
            
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
                    self.lcd.display_status("No Speech", "Try again")
                    return ""
                    
                transcript = result['results']['channels'][0]['alternatives'][0].get('transcript', '')
                
                if not transcript.strip():
                    logger.error("❌ No speech detected in the audio")
                    self.lcd.display_status("No Speech", "Try again")
                    return ""
                
                stt_time = time.time() - start_time
                logger.info(f"📝 STT ({stt_time:.2f}s): {transcript}")
                self.lcd.display_status("Heard:", transcript[:16])
                time.sleep(1)  # Brief pause to show what was heard
                return transcript
            else:
                error_msg = response.text if response.text else "No error details provided"
                logger.error(f"Deepgram error {response.status_code}: {error_msg}")
                self.lcd.display_status("STT Error", "Try again")
                return ""
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self.lcd.display_status("STT Error", str(e)[:16])
            return ""

    def get_cached_weather(self):
        """Get weather with caching for speed"""
        current_time = time.time()
        
        if (self.cached_weather is None or 
            current_time - self.last_weather_update > self.weather_cache_duration):
            
            try:
                self.lcd.display_status("Getting", "Weather...")
                url = f"http://api.openweathermap.org/data/2.5/weather?q={FastConfig.CITY_NAME}&appid={FastConfig.OPENWEATHER_API_KEY}&units=metric"
                response = requests.get(url, timeout=3)
                
                if response.status_code == 200:
                    data = response.json()
                    self.cached_weather = {
                        "temp": data["main"]["temp"],
                        "feels_like": data["main"]["feels_like"],
                        "condition": data["weather"][0]["description"],
                        "city": data["name"],
                        "humidity": data["main"]["humidity"],
                        "wind_speed": data.get("wind", {}).get("speed", 0)
                    }
                    self.last_weather_update = current_time
                    logger.info("🌤️ Weather cached")
                    
            except Exception as e:
                logger.error(f"Weather error: {e}")
                
        return self.cached_weather

    def add_to_context(self, user_input, response):
        """Add conversation to context for better responses"""
        self.conversation_context.append({
            "user": user_input,
            "assistant": response,
            "timestamp": time.time()
        })
        
        # Keep only recent context
        if len(self.conversation_context) > self.max_context_length:
            self.conversation_context.pop(0)

    def process_fast(self, question):
        """Lightning-fast question processing with LCD display"""
        start_time = time.time()
        question_lower = question.lower()
        
        try:
            # Exit commands
            if any(word in question_lower for word in ["exit", "quit", "stop", "bye", "goodbye"]):
                return "EXIT"
            
            # Greeting responses
            elif any(word in question_lower for word in FastConfig.GREETING_KEYWORDS):
                greetings = [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Hey! Ready to assist you.",
                    "Good to hear from you! What's on your mind?"
                ]
                response = greetings[int(time.time()) % len(greetings)]
                self.lcd.display_status("Hello!", "How can I help?")
            
            # Time queries
            elif any(word in question_lower for word in FastConfig.TIME_KEYWORDS):
                now = datetime.now()
                if "date" in question_lower or "day" in question_lower:
                    response = f"Today is {now.strftime('%A, %B %d, %Y')}"
                else:
                    response = f"It's {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d')}"
                
                # Show time on LCD
                self.lcd.display_time()
            
            # Weather queries
            elif any(word in question_lower for word in FastConfig.WEATHER_KEYWORDS):
                weather = self.get_cached_weather()
                if weather:
                    # Display weather on LCD
                    self.lcd.display_weather(weather)
                    
                    if "detailed" in question_lower or "details" in question_lower:
                        response = f"In {weather['city']}: {weather['temp']:.0f}°C, feels like {weather['feels_like']:.0f}°C. {weather['condition'].title()}. Humidity {weather['humidity']}%, wind {weather['wind_speed']:.1f} m/s."
                    else:
                        response = f"It's {weather['temp']:.0f} degrees with {weather['condition']} in {weather['city']}"
                else:
                    response = "Weather information unavailable right now"
                    self.lcd.display_status("Weather", "Unavailable")
            
            # LCD control commands
            elif any(word in question_lower for word in FastConfig.LCD_KEYWORDS):
                if "time" in question_lower:
                    self.lcd.display_time()
                    response = "Showing time on display"
                elif "weather" in question_lower:
                    weather = self.get_cached_weather()
                    self.lcd.display_weather(weather)
                    response = "Showing weather on display"
                elif "clear" in question_lower:
                    self.lcd.clear_display()
                    response = "Display cleared"
                else:
                    self.lcd.display_status("LCD Active", "Commands ready")
                    response = "LCD display is ready for commands"
            
            # General questions with context
            else:
                self.lcd.display_status("Thinking...", "Please wait")
                
                # Build context string
                context_str = ""
                if self.conversation_context:
                    recent_context = self.conversation_context[-2:]  # Last 2 exchanges
                    context_parts = []
                    for ctx in recent_context:
                        context_parts.append(f"User: {ctx['user']}\nAssistant: {ctx['assistant']}")
                    context_str = f"Previous conversation:\n{chr(10).join(context_parts)}\n\n"
                
                prompt = f"{context_str}Current question: {question}\n\nAnswer in exactly one short, natural sentence (maximum 20 words). Be conversational and helpful."
                
                response = gemini_model.generate_content(
                    prompt,
                    generation_config={
                        'max_output_tokens': 40,
                        'temperature': 0.3
                    }
                ).text.strip()
                
                # Show response on LCD
                words = response.split()
                if len(words) <= 4:
                    self.lcd.display_status("Response:", response[:16])
                else:
                    # Split into two lines
                    mid_point = len(words) // 2
                    line1 = ' '.join(words[:mid_point])[:16]
                    line2 = ' '.join(words[mid_point:])[:16]
                    self.lcd.display_text(line1, line2)
            
            # Add to conversation context
            self.add_to_context(question, response)
            
            process_time = time.time() - start_time
            logger.info(f"🧠 Processing ({process_time:.2f}s): {response}")
            return response
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.lcd.display_status("Error", "Processing")
            return "Sorry, I had an error processing that"

    def speak_fast(self, text):
        """Ultra-fast TTS with LCD indication"""
        start_time = time.time()
        self.lcd.indicate_speaking()
        
        try:
            # First try ElevenLabs
            if self._try_elevenlabs_tts(text):
                return
                
            # If ElevenLabs fails, try gTTS
            if self._try_gtts(text):
                return
                
            # If both TTS methods fail, fall back to text
            logger.warning("All TTS methods failed, falling back to text")
            self.lcd.display_status("TTS Failed", "Text only")
            print(f"🔊 Assistant (text only): {text}")
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            self.lcd.display_status("TTS Error", "Text only")
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
                    "similarity_boost": 0.5,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # Play the audio using pygame
                audio_data = BytesIO(response.content)
                try:
                    pygame.mixer.music.load(audio_data)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    return True
                except pygame.error as e:
                    logger.warning(f"Pygame audio error: {e}")
                    return False
            else:
                logger.warning(f"ElevenLabs TTS failed: {response.status_code} - {response.text[:200]}")
                return False
                
        except Exception as e:
            logger.warning(f"ElevenLabs TTS error: {e}")
            return False
            
    def _try_gtts(self, text):
        """Try to use gTTS, return True if successful"""
        try:
            logger.info("🔊 Trying gTTS fallback...")
            self.lcd.display_status("Using gTTS", "Fallback")
            
            # Create gTTS object
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to bytes buffer
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            # Play the audio using pygame
            try:
                pygame.mixer.music.load(audio_buffer)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                return True
            except pygame.error as e:
                logger.error(f"Pygame audio error in gTTS: {e}")
                return False
            
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            return False

    def run_continuous(self):
        """Run continuously with LCD display"""
        logger.info("🎯 Continuous mode with LCD display")
        
        # Pre-cache weather
        self.get_cached_weather()
        
        # Show ready status
        self.lcd.indicate_ready()
        
        print("""
🎙️  VOICE INTERACTION WITH LCD DISPLAY:
- LCD shows current status and responses
- Blue LED: Listening/Speaking
- White LED: Processing
- Weather and time displayed on LCD when requested
        """)
        
        try:
            while True:
                try:
                    # Wait for user input
                    self.lcd.indicate_ready()
                    user_input = input("\nPress ENTER to start recording (or 'q' to quit): ").strip().lower()
                    
                    if user_input in ['q', 'quit', 'exit']:
                        self.lcd.display_status("Goodbye!", "Shutting down")
                        logger.info("👋 Goodbye!")
                        break
                    
                    # Record and process
                    total_start = time.time()
                    
                    # Step 1: Record with adaptive timing and LCD feedback
                    audio_data = self.record_audio_adaptive()
                    if not audio_data:
                        self.lcd.display_status("No Audio", "Try again")
                        print("❌ No audio recorded")
                        continue
                    
                    # Step 2: Transcribe with LCD feedback
                    text = self.transcribe_simple(audio_data)
                    if not text or not text.strip():
                        self.lcd.display_status("No Speech", "Try again")
                        print("❌ No speech detected - try speaking louder or closer to the microphone")
                        continue
                    
                    print(f"👤 You said: {text}")
                    
                    # Step 3: Process with LCD display
                    response = self.process_fast(text)
                    if response == "EXIT":
                        self.lcd.display_status("Goodbye!", "")
                        logger.info("👋 Goodbye!")
                        break
                    
                    # Step 4: Speak with LCD indication
                    print(f"🤖 Assistant: {response}")
                    self.speak_fast(response)
                    
                    # Performance summary
                    total_time = time.time() - total_start
                    logger.info(f"⚡ Total response time: {total_time:.2f}s")
                    
                    # Show context if available
                    if len(self.conversation_context) > 1:
                        logger.info(f"💭 Context: {len(self.conversation_context)} exchanges remembered")
                    
                    # Brief pause before showing ready again
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    logger.info("👋 Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Main loop error: {e}")
                    self.lcd.display_status("Error", "Try again")
                    print("❌ An error occurred. Please try again.")
                    time.sleep(0.5)
                    
        finally:
            # Clean up LCD and GPIO
            self.lcd.cleanup()

def main():
    """Main entry point"""
    assistant = None
    try:
        assistant = UltraFastAssistant()
        
        print("""
🚀 ULTRA-FAST VOICE ASSISTANT WITH LCD v3.0
===========================================
📺 LCD FEATURES:
- 16x2 Character LCD Display
- Real-time status indication
- Weather and time display
- Visual feedback for all operations

💡 LED STATUS INDICATORS:
- Blue LED (GPIO 21): Listening/Speaking
- White LED (GPIO 20): Processing

🔌 HARDWARE CONNECTIONS:
- LCD RS: GPIO 25 (Pin 22)
- LCD Enable: GPIO 24 (Pin 18)
- LCD D4-D7: GPIO 23,17,18,22 (Pins 16,11,12,15)
- Blue LED: GPIO 21 (Pin 40)
- White LED: GPIO 20 (Pin 38)

📋 VOICE COMMANDS:
- "What's the weather?" - Shows weather on LCD
- "What time is it?" - Shows time on LCD  
- "Show weather on display" - LCD weather display
- "Show time on screen" - LCD time display
- "Clear display" - Clears LCD screen
- "Hello" - Greeting responses
        """)
        
        assistant.run_continuous()
        
    except KeyboardInterrupt:
        print("\n👋 Assistant shutdown by user")
        if assistant and assistant.lcd:
            assistant.lcd.cleanup()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        if assistant and assistant.lcd:
            assistant.lcd.cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main()
