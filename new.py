import os
import sys
import time
import threading
import queue
import logging
import json
import subprocess
import platform
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import traceback

# Core audio and ML libraries
import sounddevice as sd
import numpy as np
import wave
import whisper

# TTS Libraries (Raspberry Pi compatible)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
except ImportError:
    GTTS_AVAILABLE = False

# Raspberry Pi specific TTS
try:
    # espeak is commonly available on Raspberry Pi
    subprocess.run(['espeak', '--version'], capture_output=True, check=True)
    ESPEAK_AVAILABLE = True
except (subprocess.CalledProcessError, FileNotFoundError):
    ESPEAK_AVAILABLE = False

# Simple rule-based response system (replaces ctransformers)
import re
import random

# Configuration Constants
class Config:
    # Model Configuration
    WHISPER_MODEL = "base"  # Use base model for better Pi performance
    
    # Audio Configuration
    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = np.float32
    RECORD_DURATION = 6
    MIN_RECORDING_LENGTH = 0.5
    SILENCE_THRESHOLD = 0.01
    MAX_SILENCE_DURATION = 2.0
    
    # Simple Response System Configuration
    MAX_RESPONSE_LENGTH = 15  # words
    
    # System Configuration
    MAX_RETRY_ATTEMPTS = 3
    HEALTH_CHECK_INTERVAL = 30
    LOG_LEVEL = logging.INFO
    
    # Assistant Mode Configuration
    ASSISTANT_WAKE_WORDS = ["hello krishna", "hey krishna", "wake up krishna", "activate krishna", "krishna wake up", "wake up", "hello"]
    ASSISTANT_SLEEP_WORDS = ["sleep krishna", "shutdown", "go to sleep", "hibernate", "power down", "ok bye", "ok thank you"]
    ASSISTANT_CONTINUOUS_LISTENING_INTERVAL = 3.0
    
    # Sleep mode configuration
    SLEEP_MODE_LISTENING_INTERVAL = 5.0
    WAKE_WORD_CONFIDENCE_THRESHOLD = 0.3
    
    # Common Words
    EMERGENCY_WORDS = ["emergency", "help", "urgent", "critical"]
    MODE_SWITCH_WORDS = ["switch mode", "change mode", "test mode", "assistant mode"]

class SimpleResponseEngine:
    """Simple rule-based response engine for Raspberry Pi"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Response patterns
        self.patterns = {
            # Greetings
            r'\b(hello|hi|hey|good morning|good evening)\b': [
                "Hello! How can I help you?",
                "Hi there! What do you need?",
                "Greetings! I'm ready to assist.",
                "Hello! Krishna systems online."
            ],
            
            # Questions about time
            r'\b(time|clock|hour)\b': [
                lambda: f"Current time is {datetime.now().strftime('%I:%M %p')}."
            ],
            
            # Questions about date
            r'\b(date|day|today)\b': [
                lambda: f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."
            ],
            
            # Weather (basic response)
            r'\b(weather|temperature|hot|cold|rain)\b': [
                "I don't have weather sensors. Please check your weather app.",
                "Weather monitoring not available. Use local weather service."
            ],
            
            # System status
            r'\b(status|health|system|working)\b': [
                "All systems operational.",
                "Krishna systems functioning normally.",
                "Status: Online and ready."
            ],
            
            # Thank you
            r'\b(thank|thanks|appreciate)\b': [
                "You're welcome!",
                "Happy to help!",
                "Glad I could assist.",
                "Anytime!"
            ],
            
            # Questions about capabilities
            r'\b(what can you|help me|assist|do for me)\b': [
                "I can answer questions and help with basic tasks.",
                "I'm here to assist with information and simple requests.",
                "Ask me about time, date, or general questions."
            ],
            
            # Math operations (basic)
            r'\b(calculate|math|plus|minus|multiply|divide)\b': [
                "I can do basic math. Try asking simple calculations.",
                "For complex math, please use a calculator app."
            ],
            
            # Location/navigation
            r'\b(where|location|navigate|direction)\b': [
                "I don't have GPS capabilities. Use navigation apps.",
                "Location services not available on this system."
            ],
            
            # Entertainment
            r'\b(joke|funny|laugh|entertainment)\b': [
                "Why don't robots ever panic? They have excellent self-control!",
                "What do you call a robot who loves to garden? A plant-droid!",
                "I'd tell you a joke about UDP, but you might not get it."
            ],
            
            # Default responses for unmatched input
            'default': [
                "I'm not sure about that. Can you rephrase?",
                "Could you ask that differently?",
                "I need more information to help you.",
                "I don't understand. Please try again.",
                "That's beyond my current capabilities."
            ]
        }
    
    def generate_response(self, text: str, mode: str = "TEST") -> str:
        """Generate response based on input text and mode"""
        try:
            text_lower = text.lower().strip()
            
            # Check each pattern
            for pattern, responses in self.patterns.items():
                if pattern == 'default':
                    continue
                    
                if re.search(pattern, text_lower):
                    response_list = responses
                    if callable(response_list[0]):
                        return response_list[0]()
                    else:
                        response = random.choice(response_list)
                        # Add mode-specific personality
                        if mode == "ASSISTANT":
                            response = f"[Autonomous] {response}"
                        return response
            
            # No pattern matched, use default
            return random.choice(self.patterns['default'])
            
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            return "Processing error occurred. Please try again."

class SystemHealthMonitor:
    """Mission-critical system health monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_health_check = time.time()
        self.error_count = 0
        self.successful_interactions = 0
        self.tts_failures = 0
        self.transcription_failures = 0
        self.response_failures = 0
        self.sleep_wake_cycles = 0
        
    def log_error(self, error_type: str):
        self.error_count += 1
        if error_type == "tts":
            self.tts_failures += 1
        elif error_type == "transcription":
            self.transcription_failures += 1
        elif error_type == "response":
            self.response_failures += 1
    
    def log_success(self):
        self.successful_interactions += 1
    
    def log_sleep_wake_cycle(self):
        self.sleep_wake_cycles += 1
    
    def get_status(self) -> Dict:
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": str(timedelta(seconds=int(uptime))),
            "total_errors": self.error_count,
            "successful_interactions": self.successful_interactions,
            "success_rate": self.successful_interactions / max(1, self.successful_interactions + self.error_count),
            "tts_failures": self.tts_failures,
            "transcription_failures": self.transcription_failures,
            "response_failures": self.response_failures,
            "sleep_wake_cycles": self.sleep_wake_cycles,
            "system_status": "OPERATIONAL" if self.error_count < 10 else "DEGRADED"
        }

class RedundantTTS:
    """Multi-layer TTS system optimized for Raspberry Pi"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.primary_engine = None
        self.backup_engines = []
        self.current_engine_type = None
        self.initialize_engines()
    
    def initialize_engines(self):
        """Initialize all available TTS engines in priority order for Raspberry Pi"""
        
        # Priority 1: espeak (fast, lightweight, always available on Pi)
        if ESPEAK_AVAILABLE:
            try:
                # Test espeak
                result = subprocess.run(['espeak', '-s', '150', 'test'], 
                                        capture_output=True, timeout=5)
                self.primary_engine = ("espeak", None)
                self.current_engine_type = "espeak"
                self.logger.info("✅ Primary TTS: espeak initialized")
            except Exception as e:
                self.logger.error(f"espeak initialization failed: {e}")
        
        # Priority 2: pyttsx3 (cross-platform, offline)
        if PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)  # Slower for Pi
                engine.setProperty('volume', 0.9)
                if not self.primary_engine:
                    self.primary_engine = ("pyttsx3", engine)
                    self.current_engine_type = "pyttsx3"
                    self.logger.info("✅ Primary TTS: pyttsx3 initialized")
                else:
                    self.backup_engines.append(("pyttsx3", engine))
                    self.logger.info("✅ Backup TTS: pyttsx3 ready")
            except Exception as e:
                self.logger.error(f"pyttsx3 initialization failed: {e}")
        
        # Priority 3: gTTS (requires internet, high quality but slower)
        if GTTS_AVAILABLE:
            try:
                # Test gTTS with a quick check
                test_tts = gTTS(text="test", lang='en', slow=False)
                if not self.primary_engine:
                    self.primary_engine = ("gtts", None)
                    self.current_engine_type = "gtts"
                    self.logger.info("✅ Primary TTS: gTTS initialized")
                else:
                    self.backup_engines.append(("gtts", None))
                    self.logger.info("✅ Backup TTS: gTTS ready")
            except Exception as e:
                self.logger.error(f"gTTS initialization failed: {e}")
        
        if not self.primary_engine:
            raise RuntimeError("❌ CRITICAL: No TTS engines available!")
    
    def speak(self, text: str, timeout: float = 15.0) -> bool:
        """Speak text with automatic failover"""
        if not text or len(text.strip()) == 0:
            return False
        
        text = text.strip()
        self.logger.info(f"🗣️ Speaking: {text}")
        
        # Try primary engine first
        if self._try_speak_with_engine(self.primary_engine, text, timeout):
            return True
        
        # If primary fails, try backup engines
        for engine_info in self.backup_engines:
            self.logger.warning(f"Primary TTS failed, trying backup: {engine_info[0]}")
            if self._try_speak_with_engine(engine_info, text, timeout):
                # Promote successful backup to primary
                self.backup_engines.remove(engine_info)
                self.backup_engines.append(self.primary_engine)
                self.primary_engine = engine_info
                self.current_engine_type = engine_info[0]
                return True
        
        # All engines failed
        self.logger.error("❌ ALL TTS ENGINES FAILED")
        return False
    
    def _try_speak_with_engine(self, engine_info: Tuple, text: str, timeout: float) -> bool:
        """Try speaking with a specific engine"""
        engine_type, engine = engine_info
        
        try:
            if engine_type == "espeak":
                # Use espeak command line
                subprocess.run(['espeak', '-s', '150', '-v', 'en', text], 
                             timeout=timeout, check=True)
                return True
            
            elif engine_type == "pyttsx3":
                engine.say(text)
                engine.runAndWait()
                return True
            
            elif engine_type == "gtts":
                tts = gTTS(text=text, lang='en', slow=False)
                audio_file = f"tts_temp_{int(time.time())}.mp3"
                tts.save(audio_file)
                
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                
                # Wait for completion with timeout
                start_time = time.time()
                while pygame.mixer.music.get_busy():
                    if time.time() - start_time > timeout:
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.1)
                
                # Cleanup
                try:
                    os.remove(audio_file)
                except:
                    pass
                
                return True
        
        except Exception as e:
            self.logger.error(f"TTS engine {engine_type} failed: {e}")
            return False
        
        return False

class VoiceActivityDetector:
    """Optimized Voice Activity Detection for Raspberry Pi"""
    
    def __init__(self, sample_rate: int = Config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.frame_duration = 0.02  # 20ms frames
        self.frame_size = int(sample_rate * self.frame_duration)
        
    def detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect speech segments in audio - simplified for Pi performance"""
        # Simple energy-based VAD
        frame_count = len(audio) // self.frame_size
        energy_threshold = np.mean(np.abs(audio)) * 1.5  # Lower threshold for Pi
        
        segments = []
        in_speech = False
        speech_start = 0
        
        for i in range(frame_count):
            frame_start = i * self.frame_size
            frame_end = min(frame_start + self.frame_size, len(audio))
            frame = audio[frame_start:frame_end]
            
            frame_energy = np.mean(np.abs(frame))
            
            if frame_energy > energy_threshold and not in_speech:
                in_speech = True
                speech_start = frame_start / self.sample_rate
            elif frame_energy <= energy_threshold and in_speech:
                in_speech = False
                speech_end = frame_start / self.sample_rate
                if speech_end - speech_start > Config.MIN_RECORDING_LENGTH:
                    segments.append((speech_start, speech_end))
        
        # Handle case where speech continues to end
        if in_speech:
            segments.append((speech_start, len(audio) / self.sample_rate))
        
        return segments

class RaspberryPiVoiceAssistant:
    """Raspberry Pi optimized voice assistant"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.health_monitor = SystemHealthMonitor()
        self.tts_system = RedundantTTS()
        self.vad = VoiceActivityDetector()
        self.response_engine = SimpleResponseEngine()
        
        # Mode management
        self.current_mode = "TEST"
        self.is_assistant_awake = True
        self.is_sleeping = False
        
        # State management
        self.is_running = True
        self.continuous_listening = False
        self.last_interaction = time.time()
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Initialize AI models
        self.initialize_models()
        
        # Start background processes
        self.start_background_threads()
        
        self.logger.info("🚀 Krishna Voice Assistant - Raspberry Pi Edition - ONLINE")
    
    def setup_logging(self):
        """Configure logging for Raspberry Pi"""
        log_format = '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        # Clear any existing handlers
        logging.getLogger().handlers.clear()
        
        # Create handlers
        file_handler = logging.FileHandler('krishna_assistant.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        logging.basicConfig(
            level=Config.LOG_LEVEL,
            format=log_format,
            handlers=[file_handler, console_handler]
        )
    
    def initialize_models(self):
        """Initialize AI models optimized for Raspberry Pi"""
        self.logger.info("🧠 Initializing AI models for Raspberry Pi...")
        
        # Initialize Whisper with optimization for Pi
        try:
            # Use smaller model for better performance on Pi
            model_name = Config.WHISPER_MODEL
            self.logger.info(f"Loading Whisper model: {model_name}")
            
            self.whisper_model = whisper.load_model(model_name)
            self.logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            self.logger.critical(f"❌ Failed to load Whisper model: {e}")
            raise
        
        self.logger.info("✅ Simple response engine initialized")
    
    def start_background_threads(self):
        """Start background processing threads"""
        # Audio processing thread
        self.audio_thread = threading.Thread(target=self.process_audio_queue, daemon=True)
        self.audio_thread.start()
        
        # Response processing thread
        self.response_thread = threading.Thread(target=self.process_response_queue, daemon=True)
        self.response_thread.start()
        
        # Health monitoring thread
        self.health_thread = threading.Thread(target=self.health_monitor_loop, daemon=True)
        self.health_thread.start()
        
        # Assistant mode continuous listening thread
        self.assistant_thread = threading.Thread(target=self.assistant_mode_loop, daemon=True)
        self.assistant_thread.start()
    
    def switch_mode(self, new_mode: str):
        """Switch between Assistant and Test modes"""
        if new_mode.upper() not in ["ASSISTANT", "TEST"]:
            return False
        
        old_mode = self.current_mode
        self.current_mode = new_mode.upper()
        
        if old_mode != self.current_mode:
            self.logger.info(f"🔄 Mode switched: {old_mode} → {self.current_mode}")
            
            if self.current_mode == "ASSISTANT":
                self.continuous_listening = True
                self.is_sleeping = False
                self.tts_system.speak("Assistant mode activated. Autonomous operation started.")
            else:
                self.continuous_listening = False
                self.is_sleeping = False
                self.tts_system.speak("Test mode activated. Manual interaction enabled.")
        
        return True
    
    def enter_sleep_mode(self):
        """Enter sleep mode"""
        self.is_sleeping = True
        self.continuous_listening = False
        self.is_assistant_awake = False
        self.health_monitor.log_sleep_wake_cycle()
        self.logger.info("😴 Entering sleep mode")
        
        if self.current_mode == "ASSISTANT":
            self.tts_system.speak("Entering sleep mode. Say hello Krishna to wake me up.")
        else:
            self.tts_system.speak("System sleeping. Say hello Krishna to wake up.")
    
    def wake_up(self):
        """Wake up from sleep mode"""
        self.is_sleeping = False
        if self.current_mode == "ASSISTANT":
            self.continuous_listening = True
            self.is_assistant_awake = True
        self.logger.info("👂 Waking up")
        self.tts_system.speak("Hello! I'm awake and ready to help.")
    
    def check_for_wake_words(self, text: str) -> bool:
        """Check if text contains wake words when sleeping"""
        if not self.is_sleeping:
            return False
        
        text_lower = text.lower().strip()
        for wake_word in Config.ASSISTANT_WAKE_WORDS:
            if wake_word in text_lower:
                self.logger.info(f"👂 Wake word detected: '{wake_word}'")
                return True
        return False
    
    def assistant_mode_loop(self):
        """Continuous listening loop optimized for Raspberry Pi"""
        while self.is_running:
            try:
                if self.current_mode == "ASSISTANT":
                    if self.is_sleeping:
                        print(f"\r😴 [SLEEP] Listening for wake words... {datetime.now().strftime('%H:%M:%S')}", end="")
                        
                        audio_data = self.record_audio_sleep_mode()
                        
                        if audio_data is not None:
                            text = self.transcribe_audio(audio_data)
                            if text and self.check_for_wake_words(text):
                                self.wake_up()
                        
                        time.sleep(Config.SLEEP_MODE_LISTENING_INTERVAL)
                        
                    elif self.continuous_listening and self.is_assistant_awake:
                        print(f"\r🤖 [ASSISTANT] Listening... {datetime.now().strftime('%H:%M:%S')}", end="")
                        
                        audio_data = self.record_audio_advanced()
                        
                        if audio_data is not None:
                            self.audio_queue.put(audio_data)
                            self.last_interaction = time.time()
                        
                        time.sleep(Config.ASSISTANT_CONTINUOUS_LISTENING_INTERVAL)
                    else:
                        time.sleep(1)
                else:
                    if self.is_sleeping:
                        print(f"\r😴 [TEST-SLEEP] Say hello Krishna to wake... {datetime.now().strftime('%H:%M:%S')}", end="")
                        
                        audio_data = self.record_audio_sleep_mode()
                        if audio_data is not None:
                            text = self.transcribe_audio(audio_data)
                            if text and self.check_for_wake_words(text):
                                self.wake_up()
                        
                        time.sleep(Config.SLEEP_MODE_LISTENING_INTERVAL)
                    else:
                        time.sleep(1)
                        
            except Exception as e:
                self.logger.error(f"Assistant mode loop error: {e}")
                time.sleep(5)
    
    def record_audio_sleep_mode(self) -> Optional[np.ndarray]:
        """Audio recording for sleep mode"""
        try:
            recording = []
            
            with sd.InputStream(
                samplerate=Config.SAMPLE_RATE,
                channels=Config.CHANNELS,
                dtype=Config.DTYPE
            ) as stream:
                
                duration = 3.0  # 3 seconds for wake words
                frames = int(duration * Config.SAMPLE_RATE / 1024)
                
                for _ in range(frames):
                    chunk, overflowed = stream.read(1024)
                    chunk = chunk.flatten()
                    recording.extend(chunk)
            
            recording = np.array(recording, dtype=Config.DTYPE)
            
            if len(recording) / Config.SAMPLE_RATE < 0.5:
                return None
            
            return recording
            
        except Exception as e:
            self.logger.error(f"Sleep mode audio recording failed: {e}")
            return None
    
    def record_audio_advanced(self) -> Optional[np.ndarray]:
        """Optimized audio recording for Raspberry Pi"""
        if self.current_mode == "TEST":
            self.logger.info("🎙️ Starting audio capture...")
        
        try:
            recording = []
            silence_duration = 0
            has_speech = False
            
            with sd.InputStream(
                samplerate=Config.SAMPLE_RATE,
                channels=Config.CHANNELS,
                dtype=Config.DTYPE
            ) as stream:
                
                start_time = time.time()
                
                while time.time() - start_time < Config.RECORD_DURATION:
                    chunk, overflowed = stream.read(int(Config.SAMPLE_RATE * 0.1))
                    chunk = chunk.flatten()
                    recording.extend(chunk)
                    
                    # Check for speech
                    chunk_energy = np.mean(np.abs(chunk))
                    
                    if chunk_energy > Config.SILENCE_THRESHOLD:
                        silence_duration = 0
                        has_speech = True
                    else:
                        silence_duration += 0.1
                    
                    if has_speech and silence_duration >= Config.MAX_SILENCE_DURATION:
                        break
                    
                    # Visual feedback (only in test mode)
                    if self.current_mode == "TEST":
                        elapsed = time.time() - start_time
                        energy_bar = "█" * min(20, int(chunk_energy * 1000))
                        print(f"\r🎙️ Recording [{elapsed:.1f}s]: {energy_bar:<20}", end="")
            
            if self.current_mode == "TEST":
                print()
            
            recording = np.array(recording, dtype=Config.DTYPE)
            
            if len(recording) / Config.SAMPLE_RATE < Config.MIN_RECORDING_LENGTH:
                if self.current_mode == "TEST":
                    self.logger.warning("Recording too short")
                return None
            
            if not has_speech:
                if self.current_mode == "TEST":
                    self.logger.info("No speech detected")
                return None
            
            self.logger.info(f"✅ Audio captured: {len(recording)/Config.SAMPLE_RATE:.1f}s")
            return recording
            
        except Exception as e:
            self.logger.error(f"Audio recording failed: {e}")
            self.health_monitor.log_error("recording")
            return None
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio optimized for Raspberry Pi"""
        try:
            # Save temporary audio file
            temp_file = "temp_audio.wav"
            
            # Convert to int16 for WAV file
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(Config.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(Config.SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
            
            # Transcribe with Pi optimizations
            result = self.whisper_model.transcribe(temp_file, fp16=False, language='en')
            text = result["text"].strip()
            
            # Cleanup
            try:
                os.remove(temp_file)
            except:
                pass
            
            if len(text) < 2:
                if self.current_mode == "TEST" and not self.is_sleeping:
                    self.logger.info("Transcription too short")
                return None
            
            # Get confidence score if available
            confidence = 1.0
            if "segments" in result and result["segments"]:
                confidence = 1.0 - np.mean([seg.get("no_speech_prob", 0) for seg in result["segments"]])
            
            confidence_threshold = Config.WAKE_WORD_CONFIDENCE_THRESHOLD if self.is_sleeping else 0.3
            
            self.logger.info(f"📝 Transcribed: '{text}' (confidence: {confidence:.2f})")
            
            if confidence < confidence_threshold:
                if self.current_mode == "TEST" and not self.is_sleeping:
                    self.logger.warning("Low confidence transcription")
                return None
            
            return text
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            self.health_monitor.log_error("transcription")
            return None
    
    def process_audio_queue(self):
        """Process audio in background thread"""
        while self.is_running:
            try:
                audio_data = self.audio_queue.get(timeout=1)
                if audio_data is None:
                    continue
                
                # Transcribe audio
                text = self.transcribe_audio(audio_data)
                if text:
                    # Generate response
                    response = self.generate_response(text)
                    if response:
                        self.response_queue.put(response)
                        self.health_monitor.log_success()
                
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Audio processing error: {e}")
                self.health_monitor.log_error("processing")

    def process_response_queue(self):
        """Process responses in background thread"""
        while self.is_running:
            try:
                response = self.response_queue.get(timeout=1)
                if response is None:
                    continue
                
                if response == "SLEEP_MODE":
                    self.enter_sleep_mode()
                elif response == "WAKE_UP":
                    self.wake_up()
                elif response == "ASSISTANT_MODE_ACTIVATED":
                    continue  # Mode switch handled in switch_mode()
                elif response == "TEST_MODE_ACTIVATED":
                    continue  # Mode switch handled in switch_mode()
                else:
                    # Speak response
                    success = self.tts_system.speak(response)
                    if not success:
                        self.health_monitor.log_error("tts")
                        if self.current_mode == "TEST":
                            print(f"🔇 TTS Failed - Text: {response}")
                
                self.response_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Response processing error: {e}")
                self.health_monitor.log_error("processing")
    
    def health_monitor_loop(self):
        """Background health monitoring"""
        while self.is_running:
            try:
                time.sleep(Config.HEALTH_CHECK_INTERVAL)
                status = self.health_monitor.get_status()
                
                if status["system_status"] == "DEGRADED":
                    self.logger.warning("⚠️ System performance degraded")
                    if self.current_mode == "ASSISTANT" and not self.is_sleeping:
                        self.tts_system.speak("System performance degraded. Consider maintenance.")
                
                # Log periodic status
                sleep_status = "SLEEPING" if self.is_sleeping else "AWAKE"
                self.logger.info(f"💚 Health Check: {status['successful_interactions']} interactions, "
                                 f"{status['success_rate']:.1%} success rate, Mode: {self.current_mode}, State: {sleep_status}")
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")

    def get_pi_info(self) -> str:
        """Get Raspberry Pi information"""
        try:
            # Try to get Pi model info
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip('\x00')
            return model
        except:
            return f"ARM device (Platform: {platform.platform()})"

    def print_status(self):
        """Print current system status"""
        status = self.health_monitor.get_status()
        sleep_status = "SLEEPING" if self.is_sleeping else ("AWAKE" if self.is_assistant_awake else "STANDBY")
        listening_status = "ACTIVE" if self.continuous_listening else "INACTIVE"
        
        print(f"\n{'='*50}")
        print(f"🤖 KRISHNA STATUS REPORT")
        print(f"{'='*50}")
        print(f"Platform: Raspberry Pi")
        print(f"Current Mode: {self.current_mode}")
        print(f"System State: {sleep_status}")
        print(f"Continuous Listening: {listening_status}")
        print(f"System Status: {status['system_status']}")
        print(f"Uptime: {status['uptime_formatted']}")
        print(f"Success Rate: {status['success_rate']:.1%}")
        print(f"Sleep/Wake Cycles: {status['sleep_wake_cycles']}")
        print(f"TTS Engine: {self.tts_system.current_engine_type}")
        print(f"Successful Interactions: {status['successful_interactions']}")
        print(f"{'='*50}")

    def print_detailed_health(self):
        """Print detailed health information"""
        status = self.health_monitor.get_status()
        print(f"\n{'='*60}")
        print(f"🏥 DETAILED SYSTEM HEALTH")
        print(f"{'='*60}")
        for key, value in status.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print(f"Sleep State: {'SLEEPING' if self.is_sleeping else 'AWAKE'}")
        print(f"Platform: {self.get_pi_info()}")
        print(f"{'='*60}")
    
    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("🛑 Initiating graceful shutdown...")
        self.is_running = False
        self.continuous_listening = False
        
        # Final status report
        status = self.health_monitor.get_status()
        self.logger.info(f"Final stats: {status['successful_interactions']} interactions, "
                         f"{status['success_rate']:.1%} success rate, {status['sleep_wake_cycles']} sleep/wake cycles")
        
        shutdown_message = f"Krishna systems shutting down. Final mode: {self.current_mode}. All systems safed. Mission complete."
        self.tts_system.speak(shutdown_message)
        
        # Wait for threads to finish
        try:
            self.audio_queue.put(None)
            self.response_queue.put(None)
            time.sleep(2)
        except:
            pass
        
        self.logger.info("✅ Raspberry Pi shutdown complete")
    
    def run_dual_mode_session(self):
        """Main dual-mode session optimized for Raspberry Pi"""
        print("🚀 Krishna Voice Assistant - Raspberry Pi Edition")
        print("Modes: [a] Assistant (Autonomous) | [t] Test (Manual)")
        print("Sleep/Wake: Say 'shutdown' to sleep | Say 'hello krishna' to wake")
        print("Test Mode Commands: [ENTER] Record | [s] Status | [h] Health | [q] Quit")
        print("=" * 80)
        
        # Initial setup
        cpu_info = self.get_pi_info()
        self.logger.info(f"Running on: {cpu_info}")
        self.tts_system.speak("Krishna systems initialized on Raspberry Pi. Starting in test mode.")
        
        try:
            while self.is_running:
                if self.current_mode == "ASSISTANT":
                    # Assistant mode
                    if self.is_sleeping:
                        print(f"\n😴 ASSISTANT MODE - SLEEPING")
                        print("The assistant is sleeping and listening for 'hello krishna'")
                        print("Commands: [t] Test Mode | [w] Wake Up | [q] Quit")
                        
                        user_input = input(f"[Assistant-SLEEPING]: ").strip().lower()
                        
                        if user_input == 'q':
                            self.shutdown()
                            break
                        elif user_input == 't':
                            self.switch_mode("TEST")
                        elif user_input == 'w':
                            self.wake_up()
                        elif user_input == 's':
                            self.print_status()
                        elif user_input == 'h':
                            self.print_detailed_health()
                    else:
                        print(f"\n🤖 ASSISTANT MODE ACTIVE - Autonomous Operation")
                        print("Commands: [t] Test Mode | [sleep] Sleep | [s] Status | [q] Quit")
                        
                        user_input = input(f"[Assistant-{'LISTENING' if self.continuous_listening else 'STANDBY'}]: ").strip().lower()
                        
                        if user_input == 'q':
                            self.shutdown()
                            break
                        elif user_input == 't':
                            self.switch_mode("TEST")
                        elif user_input == 'sleep':
                            self.enter_sleep_mode()
                        elif user_input == 's':
                            self.print_status()
                        elif user_input == 'h':
                            self.print_detailed_health()
                
                else:
                    # Test mode
                    if self.is_sleeping:
                        print(f"\n😴 TEST MODE - SLEEPING")
                        print("System is sleeping and listening for 'hello krishna'")
                        print("Commands: [a] Assistant Mode | [w] Wake Up | [q] Quit")
                        
                        user_input = input("\n[Test-SLEEPING]: ").strip().lower()
                        
                        if user_input == 'q':
                            self.shutdown()
                            break
                        elif user_input == 'a':
                            self.switch_mode("ASSISTANT")
                        elif user_input == 'w':
                            self.wake_up()
                        elif user_input == 's':
                            self.print_status()
                        elif user_input == 'h':
                            self.print_detailed_health()
                    else:
                        print(f"\n🧪 TEST MODE ACTIVE - Manual Operation")
                        user_input = input("\n[ENTER] Record | [a] Assistant | [sleep] Sleep | [s] Status | [h] Health | [q] Quit: ").strip().lower()
                        
                        if user_input == 'q':
                            self.shutdown()
                            break
                        
                        elif user_input == 'a':
                            self.switch_mode("ASSISTANT")
                            continue
                        
                        elif user_input == 'sleep':
                            self.enter_sleep_mode()
                            continue
                        
                        elif user_input == 's':
                            self.print_status()
                            continue
                        
                        elif user_input == 'h':
                            self.print_detailed_health()
                            continue
                        
                        # Manual recording in test mode
                        print("\n" + "="*50)
                        audio_data = self.record_audio_advanced()
                        
                        if audio_data is not None:
                            self.audio_queue.put(audio_data)
                            self.last_interaction = time.time()
                        else:
                            print("❌ No valid audio captured")
                        
                        print("="*50)
        
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received")
            self.shutdown()
            
    def generate_response(self, text: str) -> Optional[str]:
        """Generate response using simple rule-based system"""
        try:
            text_lower = text.lower().strip()
            
            # Handle wake words when sleeping
            if self.is_sleeping and self.check_for_wake_words(text):
                return "WAKE_UP"
            
            # Don't process other commands when sleeping
            if self.is_sleeping:
                return None
            
            # Handle mode switching
            if any(phrase in text_lower for phrase in Config.MODE_SWITCH_WORDS):
                if "assistant" in text_lower:
                    if self.switch_mode("ASSISTANT"):
                        return "ASSISTANT_MODE_ACTIVATED"
                elif "test" in text_lower:
                    if self.switch_mode("TEST"):
                        return "TEST_MODE_ACTIVATED"
                else:
                    return f"Current mode: {self.current_mode}. Say assistant mode or test mode to switch."
            
            # Handle sleep commands
            if any(word in text_lower for word in Config.ASSISTANT_SLEEP_WORDS):
                return "SLEEP_MODE"
            
            # Handle emergency
            if any(word in text_lower for word in Config.EMERGENCY_WORDS):
                return "Emergency protocols activated. Please specify your emergency."
            
            # Handle status requests
            if "status" in text_lower or "health" in text_lower:
                status = self.health_monitor.get_status()
                sleep_status = "SLEEPING" if self.is_sleeping else "AWAKE"
                return f"System status: {status['system_status']}. Success rate: {status['success_rate']:.1%}. Mode: {self.current_mode}. State: {sleep_status}."
            
            # Use simple response engine
            response = self.response_engine.generate_response(text, self.current_mode)
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            return "I'm experiencing processing difficulties. Please retry."

def check_pi_dependencies():
    """Check if required dependencies are available on Pi"""
    missing_deps = []
    
    # Check core dependencies
    try:
        import sounddevice as sd
    except ImportError:
        missing_deps.append("sounddevice")
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import whisper
    except ImportError:
        missing_deps.append("openai-whisper")
    
    # Check TTS options
    tts_available = False
    
    if ESPEAK_AVAILABLE:
        tts_available = True
        print("✅ espeak TTS available")
    
    if PYTTSX3_AVAILABLE:
        tts_available = True
        print("✅ pyttsx3 TTS available")
    
    if GTTS_AVAILABLE:
        tts_available = True
        print("✅ gTTS available (requires internet)")
    
    if not tts_available:
        missing_deps.append("TTS engine (install espeak-espeak or pyttsx3)")
    
    if missing_deps:
        print("❌ Missing dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nInstall missing dependencies:")
        print("sudo apt update")
        print("sudo apt install espeak espeak-data")
        print("pip install sounddevice numpy openai-whisper pyttsx3")
        return False
    
    return True

def select_initial_mode():
    """Allow user to select initial mode"""
    print("🚀 Krishna Voice Assistant - Raspberry Pi Edition")
    print("=" * 60)
    print("Select Initial Mode:")
    print("1. TEST MODE - Manual interaction (recommended)")
    print("2. ASSISTANT MODE - Autonomous operation")
    print("\nNote: Say 'shutdown' to sleep, 'hello krishna' to wake up!")
    print("=" * 60)
    
    while True:
        try:
            choice = input("Enter choice (1 for Test, 2 for Assistant): ").strip()
            if choice == '1':
                return "TEST"
            elif choice == '2':
                return "ASSISTANT"
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n🛑 Startup cancelled")
            sys.exit(0)

def main():
    """Main entry point for Raspberry Pi"""
    try:
        print("🔧 Checking Raspberry Pi compatibility...")
        
        if not check_pi_dependencies():
            print("❌ Dependency check failed. Please install missing packages.")
            sys.exit(1)
        
        print("✅ All dependencies satisfied")
        
        # Select initial mode
        initial_mode = select_initial_mode()
        
        # Initialize assistant
        assistant = RaspberryPiVoiceAssistant()
        
        # Set initial mode
        assistant.switch_mode(initial_mode)
        
        # Run session
        assistant.run_dual_mode_session()
    
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
    
    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
