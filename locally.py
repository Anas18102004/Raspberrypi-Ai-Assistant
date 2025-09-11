import os
import time
import threading
import queue
import sounddevice as sd
import numpy as np
import wave
import pyttsx3
import whisper
from ctransformers import AutoModelForCausalLM
import logging
from datetime import datetime
import json
from pathlib import Path
import re

# ---------------- CONFIGURATION ---------------- #
class Config:
    # Paths for models
    WHISPER_MODEL_NAME = "base"
    TINYLLAMA_MODEL_PATH = r"C:\Users\moham\tinyllama\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    # Audio settings
    AUDIO_FILE = "input.wav"
    RECORD_SECONDS = 5
    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 0.01
    MIN_RECORDING_LENGTH = 1
    MAX_SILENCE_DURATION = 2.0
    
    # LLM settings
    MAX_LLM_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_P = 0.9
    MAX_RESPONSE_WORDS = 15
    
    # Wake words and sleep words
    WAKE_WORDS = ["hey robot", "ok robot", "wake up", "hey krishna", "krishna"]
    SLEEP_WORDS = ["sleep", "go to sleep", "stop listening", "goodbye"]
    
    # Conversation settings
    MAX_HISTORY = 3
    SESSION_FILE = "session_data.json"

# ---------------- LOGGING SETUP ---------------- #
def setup_logging():
    """Setup logging with file rotation"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"krishna_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ---------------- VOICE ASSISTANT CLASS ---------------- #
class KrishnaVoiceAssistant:
    def __init__(self):
        self.config = Config()
        self.conversation_history = []
        self.cache = {}
        self.robot_awake = True
        self.last_response = ""
        
        # Initialize queues
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        
        # Initialize components
        self._initialize_tts()
        self._initialize_whisper()
        self._initialize_llm()
        self._load_session_data()
    
    def _initialize_tts(self):
        """Initialize text-to-speech engine with error handling"""
        try:
            self.engine = pyttsx3.init()
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find a female voice or use the first available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
                else:
                    self.engine.setProperty('voice', voices[0].id)
            
            self.engine.setProperty('rate', 160)
            self.engine.setProperty('volume', 0.9)
            logger.info("✅ TTS engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise
    
    def _initialize_whisper(self):
        """Initialize Whisper model"""
        try:
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model(self.config.WHISPER_MODEL_NAME)
            logger.info("✅ Whisper model loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _initialize_llm(self):
        """Initialize LLM model"""
        try:
            print("Loading TinyLlama model...")
            if not os.path.exists(self.config.TINYLLAMA_MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {self.config.TINYLLAMA_MODEL_PATH}")
                
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config.TINYLLAMA_MODEL_PATH,
                max_new_tokens=self.config.MAX_LLM_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
                repetition_penalty=1.1,
                context_length=2048
            )
            logger.info("✅ TinyLlama model loaded")
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise
    
    def speak(self, text):
        """Speak text with improved error handling"""
        if not text or len(text.strip()) == 0:
            return
            
        # Clean text for better TTS
        text = self._clean_text_for_speech(text)
        
        self.last_response = text
        self.cache["last_response"] = text
        self.cache["last_response_time"] = datetime.now().isoformat()
        
        print(f"🤖 Krishna: {text}")
        logger.info(f"Speaking: {text}")
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS error: {e}")
            print(f"Failed to speak: {text}")
    
    def _clean_text_for_speech(self, text):
        """Clean text for better TTS pronunciation"""
        # Remove common problematic characters
        text = re.sub(r'[*_\[\](){}]', '', text)
        # Fix common abbreviations
        text = text.replace('&', 'and')
        text = text.replace('%', 'percent')
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def detect_silence(self, audio_data):
        """Improved silence detection"""
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms < self.config.SILENCE_THRESHOLD
    
    def record_audio_smart(self, max_duration=None):
        """Enhanced smart recording with better feedback"""
        if max_duration is None:
            max_duration = self.config.RECORD_SECONDS
            
        print("🎙️ Recording... speak now!")
        
        try:
            recording = []
            chunk_duration = 0.1
            chunk_size = int(chunk_duration * self.config.SAMPLE_RATE)
            silence_duration = 0
            has_speech = False
            
            with sd.InputStream(
                samplerate=self.config.SAMPLE_RATE,
                channels=1,
                dtype=np.float32
            ) as stream:
                start_time = time.time()
                
                while time.time() - start_time < max_duration:
                    chunk, _ = stream.read(chunk_size)
                    chunk = chunk.flatten()
                    recording.extend(chunk)
                    
                    if self.detect_silence(chunk):
                        silence_duration += chunk_duration
                    else:
                        silence_duration = 0
                        has_speech = True
                    
                    # Stop if we've had speech and then silence
                    if has_speech and silence_duration >= self.config.MAX_SILENCE_DURATION:
                        break
                    
                    # Dynamic feedback
                    elapsed = time.time() - start_time
                    status = "🔴 Speaking..." if not self.detect_silence(chunk) else "⚫ Waiting..."
                    print(f"{status} {elapsed:.1f}s", end="\r")
            
            # Process and save recording
            recording = np.array(recording, dtype=np.float32)
            duration = len(recording) / self.config.SAMPLE_RATE
            
            if duration < self.config.MIN_RECORDING_LENGTH:
                print(f"\n⚠️ Recording too short ({duration:.1f}s)")
                return False
            
            # Normalize audio
            if np.max(np.abs(recording)) > 0:
                recording = recording / np.max(np.abs(recording)) * 0.8
            
            # Save as WAV
            recording_int16 = (recording * 32767).astype(np.int16)
            with wave.open(self.config.AUDIO_FILE, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.config.SAMPLE_RATE)
                wf.writeframes(recording_int16.tobytes())
            
            print(f"\n✅ Recording saved ({duration:.1f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Recording error: {e}")
            print(f"❌ Recording failed: {e}")
            return False
    
    def transcribe(self, audio_file):
        """Enhanced transcription with confidence scoring"""
        try:
            result = self.whisper_model.transcribe(
                audio_file,
                fp16=False,
                language="en",
                temperature=0.0
            )
            text = result["text"].strip()
            
            # Calculate confidence
            if "segments" in result and result["segments"]:
                confidences = []
                for seg in result["segments"]:
                    if "no_speech_prob" in seg:
                        confidences.append(1 - seg["no_speech_prob"])
                
                if confidences:
                    avg_confidence = np.mean(confidences)
                    logger.info(f"Transcription confidence: {avg_confidence:.2f}")
                    
                    # Filter out low-confidence transcriptions
                    if avg_confidence < 0.5:
                        logger.warning("Low confidence transcription, ignoring")
                        return ""
            
            # Filter out common Whisper hallucinations
            hallucinations = [
                "thank you", "thanks for watching", "subscribe", "like and subscribe",
                "music", "applause", "silence", "quiet", ".", "?", "!"
            ]
            
            if text.lower().strip() in hallucinations:
                logger.info("Filtering out likely hallucination")
                return ""
            
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def build_context_prompt(self, current_text):
        """Build improved context-aware prompt"""
        context = ""
        if self.conversation_history:
            context = "Previous conversation:\n"
            for exchange in self.conversation_history[-2:]:  # Last 2 exchanges only
                context += f"Human: {exchange['human']}\nKrishna: {exchange['assistant']}\n"
            context += "\n"
        
        prompt = f"""You are Krishna, a helpful robot assistant. Keep responses very short (1 sentence, max 15 words).
Be natural, friendly, and conversational. Don't repeat the human's question.

{context}Human: {current_text}
Krishna:"""
        
        return prompt
    
    def handle_intent(self, text):
        """Enhanced intent handling with better responses"""
        text_lower = text.lower().strip()
        
        # Ignore very short or empty transcriptions
        if len(text.strip()) < 2:
            return None

        # Sleep/Wake control
        if any(word in text_lower for word in self.config.SLEEP_WORDS):
            self.robot_awake = False
            return "Going to sleep. Say 'hey Krishna' to wake me up."
        
        if any(word in text_lower for word in self.config.WAKE_WORDS):
            if not self.robot_awake:
                self.robot_awake = True
                return "Hello! I'm Krishna. How can I help?"
            else:
                return "I'm here! What can I do for you?"

        if not self.robot_awake:
            return None

        # Built-in commands
        if any(word in text_lower for word in ["time", "what time"]):
            current_time = datetime.now().strftime("%I:%M %p")
            return f"It's {current_time}."
        
        if "date" in text_lower or "today" in text_lower:
            current_date = datetime.now().strftime("%A, %B %d")
            return f"Today is {current_date}."
        
        if "weather" in text_lower:
            return "I can't check weather, but try your weather app."
        
        if "repeat" in text_lower or "say that again" in text_lower:
            return self.cache.get("last_response", "I have nothing to repeat.")
        
        if any(word in text_lower for word in ["exit", "quit", "shutdown"]):
            return "EXIT"
        
        if "clear history" in text_lower or "forget" in text_lower:
            self.conversation_history.clear()
            return "I've cleared our conversation history."
        
        if "help" in text_lower or "what can you do" in text_lower:
            return "I can chat, tell time and date, or answer questions. Just talk to me!"

        # Use LLM for general conversation
        return self._get_llm_response(text)
    
    def _get_llm_response(self, text):
        """Get response from LLM with improved processing"""
        try:
            prompt = self.build_context_prompt(text)
            logger.info(f"Prompt length: {len(prompt)} characters")
            
            # Generate response
            response = self.llm(prompt, max_new_tokens=50, stop=["Human:", "\n\n"])
            
            if isinstance(response, str):
                response = self._clean_llm_response(response)
                
                # Add to conversation history
                if len(response) > 2:
                    self.conversation_history.append({
                        "human": text,
                        "assistant": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Keep history manageable
                    if len(self.conversation_history) > self.config.MAX_HISTORY:
                        self.conversation_history = self.conversation_history[-self.config.MAX_HISTORY:]
                
                return response if len(response) > 2 else "I'm not sure I understand."
                
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, I had a processing error."

        return "I don't understand that."
    
    def _clean_llm_response(self, response):
        """Clean and format LLM response"""
        # Remove common prefixes
        prefixes = ["Krishna:", "Assistant:", "AI:", "Robot:", "Human:", "I think", "Well,"]
        response = response.strip()
        
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Take first sentence only
        sentences = re.split(r'[.!?]+', response)
        response = sentences[0].strip()
        
        # Limit word count
        words = response.split()
        if len(words) > self.config.MAX_RESPONSE_WORDS:
            response = ' '.join(words[:self.config.MAX_RESPONSE_WORDS])
        
        # Add punctuation if missing
        if response and response[-1] not in '.!?':
            response += '.'
        
        # Remove any remaining problematic patterns
        response = re.sub(r'\b(um|uh|like|you know)\b', '', response, flags=re.IGNORECASE)
        response = ' '.join(response.split())  # Clean whitespace
        
        return response
    
    def _save_session_data(self):
        """Save session data with error handling"""
        try:
            session_data = {
                "conversation_history": self.conversation_history,
                "cache": self.cache,
                "robot_awake": self.robot_awake,
                "timestamp": datetime.now().isoformat()
            }
            with open(self.config.SESSION_FILE, "w") as f:
                json.dump(session_data, f, indent=2)
            logger.info("Session data saved")
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
    
    def _load_session_data(self):
        """Load session data with error handling"""
        try:
            if os.path.exists(self.config.SESSION_FILE):
                with open(self.config.SESSION_FILE, "r") as f:
                    session_data = json.load(f)
                self.conversation_history = session_data.get("conversation_history", [])
                self.cache.update(session_data.get("cache", {}))
                self.robot_awake = session_data.get("robot_awake", True)
                logger.info("Session data loaded")
        except Exception as e:
            logger.error(f"Failed to load session data: {e}")
    
    def process_audio_worker(self):
        """Background audio processing"""
        while True:
            audio_file = self.audio_queue.get()
            if audio_file is None:
                break
            
            transcription = self.transcribe(audio_file)
            if transcription and len(transcription.strip()) > 1:
                print(f"🗣️ You said: '{transcription}'")
                logger.info(f"Transcribed: {transcription}")
                
                response = self.handle_intent(transcription)
                if response:
                    self.text_queue.put(response)
            else:
                logger.info("No clear speech detected")
                
            self.audio_queue.task_done()
    
    def process_text_worker(self):
        """Background text-to-speech processing"""
        while True:
            text = self.text_queue.get()
            if text is None:
                break
            if text == "EXIT":
                print("👋 Krishna is shutting down...")
                self._save_session_data()
                os._exit(0)
            self.speak(text)
            self.text_queue.task_done()
    
    def run(self):
        """Main application loop"""
        print("🤖 Krishna Robot Assistant v2.0 Online")
        print("Commands: [ENTER] Record | [s] Status | [t] Test TTS | [q] Quit")
        
        # Start background threads
        threading.Thread(target=self.process_audio_worker, daemon=True).start()
        threading.Thread(target=self.process_text_worker, daemon=True).start()
        
        # Initial greeting
        if self.robot_awake:
            self.speak("Hello! I'm Krishna, your robot assistant. Ready to help!")

        try:
            while True:
                user_input = input("\n[ENTER] Record | [s] Status | [t] Test | [q] Quit: ").strip().lower()
                
                if user_input == "q":
                    print("Shutting down Krishna...")
                    break
                elif user_input == "s":
                    status = "awake" if self.robot_awake else "sleeping"
                    print(f"Status: Krishna is {status}")
                    print(f"Conversation history: {len(self.conversation_history)} exchanges")
                    if self.last_response:
                        print(f"Last response: '{self.last_response}'")
                elif user_input == "t":
                    self.speak("This is a test of the text to speech system.")
                elif user_input == "":  # Enter key
                    if self.record_audio_smart():
                        self.audio_queue.put(self.config.AUDIO_FILE)
                    else:
                        print("❌ Recording failed or too short. Please try again.")

        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received. Shutting down...")
        finally:
            self._save_session_data()
            # Cleanup queues
            self.audio_queue.put(None)
            self.text_queue.put(None)

# ---------------- MAIN EXECUTION ---------------- #
if __name__ == "__main__":
    try:
        assistant = KrishnaVoiceAssistant()
        assistant.run()
    except Exception as e:
        logger.error(f"Failed to start Krishna: {e}")
        print(f"❌ Startup error: {e}")
        input("Press Enter to exit...")
