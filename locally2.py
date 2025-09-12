
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
from ctransformers import AutoModelForCausalLM

# TTS Libraries (multiple redundancy layers)
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

try:
    import win32com.client
    SAPI_AVAILABLE = True
except ImportError:
    SAPI_AVAILABLE = False

# Configuration Constants
class Config:
    # Model Configuration
    WHISPER_MODEL = "base"
    TINYLLAMA_PATH = r"C:\Users\moham\tinyllama\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    # Audio Configuration
    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = np.float32
    RECORD_DURATION = 6
    MIN_RECORDING_LENGTH = 0.5
    SILENCE_THRESHOLD = 0.01
    MAX_SILENCE_DURATION = 2.0
    
    # LLM Configuration
    MAX_TOKENS = 25
    TEMPERATURE = 0.1
    TOP_P = 0.8
    CONTEXT_LENGTH = 256
    REPETITION_PENALTY = 1.3
    
    # System Configuration
    MAX_RETRY_ATTEMPTS = 3
    HEALTH_CHECK_INTERVAL = 30
    LOG_LEVEL = logging.INFO
    
    # Assistant Mode Configuration
    ASSISTANT_WAKE_WORDS = ["hello krishna", "hey krishna", "wake up krishna", "activate krishna", "krishna wake up", "wake up krishna","wake up ","hello"]
    ASSISTANT_SLEEP_WORDS = ["sleep krishna", "shutdown", "go to sleep", "hibernate", "power down","ok bye","ok thank you"]
    ASSISTANT_CONTINUOUS_LISTENING_INTERVAL = 3.0  # Seconds between listening cycles
    
    # Sleep mode configuration
    SLEEP_MODE_LISTENING_INTERVAL = 5.0  # Longer interval when sleeping
    WAKE_WORD_CONFIDENCE_THRESHOLD = 0.3
    
    # Common Words
    EMERGENCY_WORDS = ["emergency", "help", "urgent", "critical"]
    MODE_SWITCH_WORDS = ["switch mode", "change mode", "test mode", "assistant mode"]

class SystemHealthMonitor:
    """Mission-critical system health monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_health_check = time.time()
        self.error_count = 0
        self.successful_interactions = 0
        self.tts_failures = 0
        self.transcription_failures = 0
        self.llm_failures = 0
        self.sleep_wake_cycles = 0
        
    def log_error(self, error_type: str):
        self.error_count += 1
        if error_type == "tts":
            self.tts_failures += 1
        elif error_type == "transcription":
            self.transcription_failures += 1
        elif error_type == "llm":
            self.llm_failures += 1
    
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
            "llm_failures": self.llm_failures,
            "sleep_wake_cycles": self.sleep_wake_cycles,
            "system_status": "OPERATIONAL" if self.error_count < 10 else "DEGRADED"
        }

class RedundantTTS:
    """Multi-layer TTS system with automatic failover"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.primary_engine = None
        self.backup_engines = []
        self.current_engine_type = None
        self.initialize_engines()
    
    def initialize_engines(self):
        """Initialize all available TTS engines in priority order"""
        
        # Priority 1: Windows SAPI (fastest, most reliable for Windows)
        if SAPI_AVAILABLE and platform.system() == "Windows":
            try:
                engine = win32com.client.Dispatch("SAPI.SpVoice")
                self.primary_engine = ("sapi", engine)
                self.current_engine_type = "sapi"
                self.logger.info("✅ Primary TTS: Windows SAPI initialized")
            except Exception as e:
                self.logger.error(f"SAPI initialization failed: {e}")
        
        # Priority 2: pyttsx3 (cross-platform, offline)
        if PYTTSX3_AVAILABLE:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)
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
        
        # Priority 3: gTTS (requires internet, high quality)
        if GTTS_AVAILABLE:
            try:
                # Test gTTS with a dummy call
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
    
    def speak(self, text: str, timeout: float = 10.0) -> bool:
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
            if engine_type == "sapi":
                engine.Speak(text)
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
    """Advanced Voice Activity Detection"""
    
    def __init__(self, sample_rate: int = Config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.frame_duration = 0.02  # 20ms frames
        self.frame_size = int(sample_rate * self.frame_duration)
        
    def detect_speech_segments(self, audio: np.ndarray) -> List[Tuple[float, float]]:
        """Detect speech segments in audio"""
        # Simple energy-based VAD
        frame_count = len(audio) // self.frame_size
        energy_threshold = np.mean(np.abs(audio)) * 2
        
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

class AerospaceVoiceAssistant:
    """Main voice assistant class with aerospace-grade reliability and dual modes"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.health_monitor = SystemHealthMonitor()
        self.tts_system = RedundantTTS()
        self.vad = VoiceActivityDetector()
        
        # Mode management
        self.current_mode = "TEST"  # Start in test mode
        self.is_assistant_awake = True  # For assistant mode sleep/wake
        self.is_sleeping = False  # New: Sleep state for shutdown/wake cycle
        
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
        
        self.logger.info("🚀 Krishna Voice Assistant - Aerospace Grade - DUAL MODE - ONLINE")
    
    def setup_logging(self):
        """Configure comprehensive logging with Unicode support"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        # Clear any existing handlers
        logging.getLogger().handlers.clear()
        
        # Create handlers with UTF-8 encoding
        file_handler = logging.FileHandler('krishna_assistant.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # For Windows, set the console to handle UTF-8
        if platform.system() == "Windows":
            try:
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
            except:
                # Fallback: Remove emojis from console output for Windows
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.addFilter(self._emoji_filter)
                console_handler.setFormatter(logging.Formatter(log_format))
        
        logging.basicConfig(
            level=Config.LOG_LEVEL,
            format=log_format,
            handlers=[file_handler, console_handler]
        )
    
    def _emoji_filter(self, record):
        """Filter to remove emojis for Windows console compatibility"""
        # Replace common emojis with text equivalents
        emoji_replacements = {
            '✅': '[OK]',
            '❌': '[ERROR]',
            '📝': '[TRANSCRIBE]',
            '🗣️': '[SPEAK]',
            '💚': '[HEALTH]',
            '🎙️': '[MIC]',
            '🤖': '[ROBOT]',
            '⚠️': '[WARNING]',
            '🛑': '[STOP]',
            '🚀': '[START]',
            '🧠': '[AI]',
            '💥': '[CRITICAL]',
            '🔇': '[MUTE]',
            '😴': '[SLEEP]',
            '👂': '[LISTEN]'
        }
        
        message = record.getMessage()
        for emoji, replacement in emoji_replacements.items():
            message = message.replace(emoji, replacement)
        record.msg = message
        record.args = ()
        return True
    
    def initialize_models(self):
        """Initialize AI models with error handling"""
        self.logger.info("🧠 Initializing AI models...")
        
        # Initialize Whisper
        try:
            self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
            self.logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            self.logger.critical(f"❌ Failed to load Whisper model: {e}")
            raise
        
        # Initialize TinyLlama
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                Config.TINYLLAMA_PATH,
                max_new_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                context_length=Config.CONTEXT_LENGTH,
                repetition_penalty=Config.REPETITION_PENALTY,
                stop=["Human:", "User:", "Krishna:", "\n", ".", "!", "?"]
            )
            self.logger.info("✅ TinyLlama model loaded successfully")
        except Exception as e:
            self.logger.critical(f"❌ Failed to load TinyLlama model: {e}")
            raise
    
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
                self.tts_system.speak("Assistant mode activated. Autonomous operation initiated. I am now listening continuously.")
            else:
                self.continuous_listening = False
                self.is_sleeping = False
                self.tts_system.speak("Test mode activated. Manual interaction mode enabled.")
        
        return True
    
    def enter_sleep_mode(self):
        """Enter sleep mode - stops active listening but keeps wake word detection"""
        self.is_sleeping = True
        self.continuous_listening = False
        self.is_assistant_awake = False
        self.health_monitor.log_sleep_wake_cycle()
        self.logger.info("😴 Entering sleep mode - Wake word detection active")
        
        if self.current_mode == "ASSISTANT":
            self.tts_system.speak("Entering sleep mode. Say 'Hello Krishna' to wake me up.")
        else:
            self.tts_system.speak("System sleeping. Say 'Hello Krishna' to wake up.")
    
    def wake_up(self):
        """Wake up from sleep mode"""
        self.is_sleeping = False
        if self.current_mode == "ASSISTANT":
            self.continuous_listening = True
            self.is_assistant_awake = True
        self.logger.info("👂 Waking up - Resuming normal operation")
        self.tts_system.speak("Hello! I'm awake and ready to assist you.")
    
    def check_for_wake_words(self, text: str) -> bool:
        """Check if text contains wake words when sleeping"""
        if not self.is_sleeping:
            return False
        
        text_lower = text.lower().strip()
        for wake_word in Config.ASSISTANT_WAKE_WORDS:
            if wake_word in text_lower:
                self.logger.info(f"👂 Wake word detected: '{wake_word}' in '{text}'")
                return True
        return False
    
    def assistant_mode_loop(self):
        """Continuous listening loop for Assistant mode with sleep/wake support"""
        while self.is_running:
            try:
                if self.current_mode == "ASSISTANT":
                    if self.is_sleeping:
                        # Sleep mode - only listen for wake words
                        print(f"\r😴 [SLEEP MODE] Listening for wake words... {datetime.now().strftime('%H:%M:%S')}", end="")
                        
                        # Record audio with longer intervals
                        audio_data = self.record_audio_sleep_mode()
                        
                        if audio_data is not None:
                            # Only check for wake words
                            text = self.transcribe_audio(audio_data)
                            if text and self.check_for_wake_words(text):
                                self.wake_up()
                        
                        time.sleep(Config.SLEEP_MODE_LISTENING_INTERVAL)
                        
                    elif self.continuous_listening and self.is_assistant_awake:
                        # Normal assistant mode
                        print(f"\r🤖 [ASSISTANT MODE] Listening... {datetime.now().strftime('%H:%M:%S')}", end="")
                        
                        # Record audio automatically
                        audio_data = self.record_audio_advanced()
                        
                        if audio_data is not None:
                            self.audio_queue.put(audio_data)
                            self.last_interaction = time.time()
                        
                        # Wait before next listening cycle
                        time.sleep(Config.ASSISTANT_CONTINUOUS_LISTENING_INTERVAL)
                    else:
                        time.sleep(1)  # Sleep when not listening
                else:
                    # Test mode - check for sleep mode
                    if self.is_sleeping:
                        print(f"\r😴 [TEST MODE - SLEEPING] Say 'Hello Krishna' to wake... {datetime.now().strftime('%H:%M:%S')}", end="")
                        
                        audio_data = self.record_audio_sleep_mode()
                        if audio_data is not None:
                            text = self.transcribe_audio(audio_data)
                            if text and self.check_for_wake_words(text):
                                self.wake_up()
                        
                        time.sleep(Config.SLEEP_MODE_LISTENING_INTERVAL)
                    else:
                        time.sleep(1)  # Normal test mode - no continuous listening
                    
            except Exception as e:
                self.logger.error(f"Assistant mode loop error: {e}")
                time.sleep(5)  # Longer sleep on error
    
    def record_audio_sleep_mode(self) -> Optional[np.ndarray]:
        """Simplified audio recording for sleep mode wake word detection"""
        try:
            # Shorter recording for wake word detection
            recording = []
            
            with sd.InputStream(
                samplerate=Config.SAMPLE_RATE,
                channels=Config.CHANNELS,
                dtype=Config.DTYPE
            ) as stream:
                
                # Record for shorter duration in sleep mode
                duration = 3.0  # 3 seconds for wake words
                frames = int(duration * Config.SAMPLE_RATE / 1024)
                
                for _ in range(frames):
                    chunk, overflowed = stream.read(1024)
                    chunk = chunk.flatten()
                    recording.extend(chunk)
                    
                    # Check for basic voice activity
                    chunk_energy = np.mean(np.abs(chunk))
                    if chunk_energy > Config.SILENCE_THRESHOLD:
                        # Continue recording if voice detected
                        continue
            
            recording = np.array(recording, dtype=Config.DTYPE)
            
            # Basic validation
            if len(recording) / Config.SAMPLE_RATE < 0.5:
                return None
            
            return recording
            
        except Exception as e:
            self.logger.error(f"Sleep mode audio recording failed: {e}")
            return None
    
    def record_audio_advanced(self) -> Optional[np.ndarray]:
        """Advanced audio recording with VAD"""
        if self.current_mode == "TEST":
            self.logger.info("🎙️ Starting manual audio capture...")
        
        try:
            # Dynamic recording with voice activity detection
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
                    # Read audio chunk
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
                    
                    # Stop if we have speech and then sufficient silence
                    if has_speech and silence_duration >= Config.MAX_SILENCE_DURATION:
                        break
                    
                    # Visual feedback (only in test mode)
                    if self.current_mode == "TEST":
                        elapsed = time.time() - start_time
                        energy_bar = "█" * min(20, int(chunk_energy * 1000))
                        print(f"\r🎙️ Recording [{elapsed:.1f}s]: {energy_bar:<20}", end="")
            
            if self.current_mode == "TEST":
                print()  # New line after recording
            
            recording = np.array(recording, dtype=Config.DTYPE)
            
            # Validate recording
            if len(recording) / Config.SAMPLE_RATE < Config.MIN_RECORDING_LENGTH:
                if self.current_mode == "TEST":
                    self.logger.warning("Recording too short")
                return None
            
            if not has_speech:
                if self.current_mode == "TEST":
                    self.logger.info("No speech detected in recording")
                return None
            
            self.logger.info(f"✅ Audio captured: {len(recording)/Config.SAMPLE_RATE:.1f}s")
            return recording
            
        except Exception as e:
            self.logger.error(f"Audio recording failed: {e}")
            self.health_monitor.log_error("recording")
            return None
    
    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio with error handling and validation"""
        try:
            # Save temporary audio file
            temp_file = "temp_audio.wav"
            
            # Convert to int16 for WAV file
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(Config.CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(Config.SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
            
            # Transcribe
            result = self.whisper_model.transcribe(temp_file, fp16=False)
            text = result["text"].strip()
            
            # Cleanup
            try:
                os.remove(temp_file)
            except:
                pass
            
            # Validate transcription
            if len(text) < 2:
                if self.current_mode == "TEST" and not self.is_sleeping:
                    self.logger.info("Transcription too short")
                return None
            
            # Get confidence score if available
            confidence = 1.0
            if "segments" in result and result["segments"]:
                confidence = 1.0 - np.mean([seg.get("no_speech_prob", 0) for seg in result["segments"]])
            
            # Lower confidence threshold for wake words when sleeping
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
    
    def generate_response(self, text: str) -> Optional[str]:
        """Generate intelligent response with sleep/wake cycle support"""
        try:
            text_lower = text.lower().strip()
            
            # Handle wake words when sleeping
            if self.is_sleeping and self.check_for_wake_words(text):
                return "WAKE_UP"
            
            # Don't process other commands when sleeping (except wake words)
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
                    return f"Current mode: {self.current_mode}. Say 'assistant mode' or 'test mode' to switch."
            
            # Handle sleep commands
            if any(word in text_lower for word in Config.ASSISTANT_SLEEP_WORDS):
                return "SLEEP_MODE"
            
            # Handle wake/sleep commands (legacy support)
            if self.current_mode == "ASSISTANT":
                if any(word in text_lower for word in Config.ASSISTANT_WAKE_WORDS):
                    if not self.is_assistant_awake:
                        self.is_assistant_awake = True
                        self.continuous_listening = True
                        return "Krishna systems reactivated. Autonomous listening resumed."
                    return "I am already active and listening."
            
            # Emergency handling
            if any(word in text_lower for word in Config.EMERGENCY_WORDS):
                return "Emergency protocols activated. Please specify the nature of your emergency."
            
            # Built-in commands
            if "time" in text_lower:
                return f"Current time is {datetime.now().strftime('%I:%M %p')}."
            
            if "date" in text_lower:
                return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."
            
            if "status" in text_lower or "health" in text_lower:
                status = self.health_monitor.get_status()
                sleep_status = "SLEEPING" if self.is_sleeping else "AWAKE"
                return f"System status: {status['system_status']}. Success rate: {status['success_rate']:.1%}. Mode: {self.current_mode}. State: {sleep_status}."
            
            if "mode" in text_lower and "current" in text_lower:
                awake_status = "SLEEPING" if self.is_sleeping else ("AWAKE" if self.is_assistant_awake else "STANDBY")
                return f"Current mode: {self.current_mode}. Assistant status: {awake_status}."
            
            # LLM generation with mode-aware personality
            if self.current_mode == "ASSISTANT":
                prompt = f"Krishna is an autonomous aerospace robot assistant like those used by NASA and ISRO. Respond professionally in 8 words or less.\nHuman: {text}\nKrishna:"
            else:
                prompt = f"Krishna is a helpful test assistant. Answer concisely in 8 words or less.\nHuman: {text}\nKrishna:"
            
            response = self.llm(prompt)
            
            if isinstance(response, str):
                # Clean and validate response
                response = response.strip()
                
                # Remove unwanted prefixes
                for prefix in ["Krishna:", "Assistant:", "AI:", "Human:"]:
                    if response.startswith(prefix):
                        response = response[len(prefix):].strip()
                
                # Limit word count
                words = response.split()[:8]
                response = ' '.join(words)
                
                # Add punctuation
                if response and response[-1] not in '.!?':
                    response += '.'
                
                # Validate response quality
                if len(response) < 3 or response.count(' ') > 10:
                    return "I need more information to assist you."
                
                return response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            self.health_monitor.log_error("llm")
            return "I'm experiencing processing difficulties. Please retry."
        
        return "I don't understand. Could you rephrase that?"
    
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
    
    def run_dual_mode_session(self):
        """Main dual-mode session with sleep/wake support"""
        print("🚀 Krishna Voice Assistant - Aerospace Grade - DUAL MODE with SLEEP/WAKE")
        print("Modes: [a] Assistant (Autonomous) | [t] Test (Manual)")
        print("Sleep/Wake: Say 'shutdown' to sleep | Say 'hello krishna' to wake")
        print("Test Mode Commands: [ENTER] Record | [s] Status | [h] Health | [q] Quit")
        print("=" * 80)
        
        # Initial setup
        self.tts_system.speak("Krishna systems initialized. Starting in test mode. Say 'shutdown' to sleep or 'hello krishna' to wake up.")
        
        try:
            while self.is_running:
                if self.current_mode == "ASSISTANT":
                    # Assistant mode - autonomous operation
                    if self.is_sleeping:
                        print(f"\n😴 ASSISTANT MODE - SLEEPING")
                        print("The assistant is sleeping and listening for 'hello krishna' to wake up.")
                        print("Commands: [t] Switch to Test Mode | [w] Wake Up | [q] Quit")
                        
                        user_input = input(f"[Assistant Mode - SLEEPING]: ").strip().lower()
                        
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
                        print("Commands: [t] Switch to Test Mode | [sleep] Put to Sleep | [s] Status | [q] Quit")
                        
                        user_input = input(f"[Assistant Mode - {'LISTENING' if self.continuous_listening else 'STANDBY'}]: ").strip().lower()
                        
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
                    # Test mode - manual operation
                    if self.is_sleeping:
                        print(f"\n😴 TEST MODE - SLEEPING")
                        print("The system is sleeping and listening for 'hello krishna' to wake up.")
                        print("Commands: [a] Assistant Mode | [w] Wake Up | [q] Quit")
                        
                        user_input = input("\n[Test Mode - SLEEPING]: ").strip().lower()
                        
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
                        user_input = input("\n[ENTER] Record | [a] Assistant Mode | [sleep] Sleep | [s] Status | [h] Health | [q] Quit: ").strip().lower()
                        
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
    
    def print_status(self):
        """Print current system status"""
        status = self.health_monitor.get_status()
        sleep_status = "SLEEPING" if self.is_sleeping else ("AWAKE" if self.is_assistant_awake else "STANDBY")
        listening_status = "ACTIVE" if self.continuous_listening else "INACTIVE"
        
        print(f"\n{'='*50}")
        print(f"🤖 KRISHNA STATUS REPORT")
        print(f"{'='*50}")
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
        print(f"{'='*60}")
    
    def shutdown(self):
        """Graceful shutdown - completely terminates the program"""
        self.logger.info("🛑 Initiating graceful shutdown...")
        self.is_running = False
        self.continuous_listening = False
        
        # Final status report
        status = self.health_monitor.get_status()
        self.logger.info(f"Final stats: {status['successful_interactions']} interactions, "
                        f"{status['success_rate']:.1%} success rate, {status['sleep_wake_cycles']} sleep/wake cycles")
        
        shutdown_message = f"Krishna systems shutting down completely. Final mode: {self.current_mode}. All systems safed. Mission complete."
        self.tts_system.speak(shutdown_message)
        
        # Wait for threads to finish
        try:
            self.audio_queue.put(None)
            self.response_queue.put(None)
            time.sleep(2)  # Allow final processing
        except:
            pass
        
        self.logger.info("✅ Complete shutdown finished")

def select_initial_mode():
    """Allow user to select initial mode"""
    print("🚀 Krishna Voice Assistant - Aerospace Grade with Sleep/Wake")
    print("=" * 60)
    print("Select Initial Mode:")
    print("1. TEST MODE - Manual interaction (recommended for testing)")
    print("2. ASSISTANT MODE - Autonomous operation (NASA/ISRO style)")
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
    """Main entry point with mode selection"""
    try:
        # Select initial mode
        initial_mode = select_initial_mode()
        
        # Initialize assistant
        assistant = AerospaceVoiceAssistant()
        
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
