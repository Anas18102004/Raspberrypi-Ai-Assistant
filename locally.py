import os
import time
import threading
import queue
import sounddevice as sd
import numpy as np
import wave
import whisper
import requests
import json
import logging
from datetime import datetime
from pathlib import Path
import re

# ---------------- CONFIGURATION ---------------- #
class Config:
    # Paths for models
    WHISPER_MODEL_NAME = "base"
    
    # Ollama settings
    OLLAMA_MODEL = "tinyllama"  # Make sure this matches your installed model name
    OLLAMA_URL = "http://localhost:11434/api/generate"
    
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
        self._initialize_whisper()
        self._test_ollama_connection()
        self._load_session_data()
    
    # ---------------- TTS USING PICO ---------------- #
    def speak(self, text):
        if not text or len(text.strip()) == 0:
            return
        
        text = self._clean_text_for_speech(text)
        self.last_response = text
        self.cache["last_response"] = text
        self.cache["last_response_time"] = datetime.now().isoformat()
        
        print(f"🤖 Krishna: {text}")
        logger.info(f"Speaking: {text}")
        
        try:
            # Use PicoTTS to generate and play speech
            os.system(f'pico2wave -w temp.wav "{text}" && aplay temp.wav')
        except Exception as e:
            logger.error(f"TTS error: {e}")
            print(f"Failed to speak: {text}")
    
    def _clean_text_for_speech(self, text):
        text = re.sub(r'[*_\[\](){}]', '', text)
        text = text.replace('&', 'and').replace('%', 'percent')
        text = ' '.join(text.split())
        return text
    
    # ---------------- WHISPER ---------------- #
    def _initialize_whisper(self):
        try:
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model(self.config.WHISPER_MODEL_NAME)
            logger.info("✔ Whisper model loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def detect_silence(self, audio_data):
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms < self.config.SILENCE_THRESHOLD
    
    def record_audio_smart(self, max_duration=None):
        if max_duration is None:
            max_duration = self.config.RECORD_SECONDS
            
        print("🎤 Recording... speak now!")
        try:
            recording = []
            chunk_duration = 0.1
            chunk_size = int(chunk_duration * self.config.SAMPLE_RATE)
            silence_duration = 0
            has_speech = False
            
            with sd.InputStream(samplerate=self.config.SAMPLE_RATE, channels=1, dtype=np.float32) as stream:
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
                    
                    if has_speech and silence_duration >= self.config.MAX_SILENCE_DURATION:
                        break
                    
                    elapsed = time.time() - start_time
                    status = "🗣 Speaking..." if not self.detect_silence(chunk) else "💤 Waiting..."
                    print(f"{status} {elapsed:.1f}s", end="\r")
            
            recording = np.array(recording, dtype=np.float32)
            duration = len(recording) / self.config.SAMPLE_RATE
            
            if duration < self.config.MIN_RECORDING_LENGTH:
                print(f"\n⚠ Recording too short ({duration:.1f}s)")
                return False
            
            if np.max(np.abs(recording)) > 0:
                recording = recording / np.max(np.abs(recording)) * 0.8
            
            recording_int16 = (recording * 32767).astype(np.int16)
            with wave.open(self.config.AUDIO_FILE, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.config.SAMPLE_RATE)
                wf.writeframes(recording_int16.tobytes())
            
            print(f"\n✔ Recording saved ({duration:.1f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Recording error: {e}")
            print(f"❌ Recording failed: {e}")
            return False
    
    def transcribe(self, audio_file):
        try:
            result = self.whisper_model.transcribe(audio_file, fp16=False, language="en", temperature=0.0)
            text = result["text"].strip()
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    # ---------------- OLLAMA INTEGRATION ---------------- #
    def _test_ollama_connection(self):
        """Test if Ollama is running and the model is available"""
        try:
            print("Testing Ollama connection...")
            
            # Test if Ollama server is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama server not responding")
            
            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if not any(self.config.OLLAMA_MODEL in name for name in model_names):
                print(f"Available models: {model_names}")
                raise Exception(f"Model '{self.config.OLLAMA_MODEL}' not found in Ollama")
            
            logger.info(f"✔ Ollama connection successful, model '{self.config.OLLAMA_MODEL}' available")
            print(f"✔ Connected to Ollama with model: {self.config.OLLAMA_MODEL}")
            
        except requests.exceptions.ConnectionError:
            logger.error("Ollama server is not running. Please start Ollama with 'ollama serve'")
            raise Exception("Ollama server not running. Run 'ollama serve' first.")
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            raise
    
    def _call_ollama(self, prompt):
        """Make API call to Ollama"""
        try:
            payload = {
                "model": self.config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.TEMPERATURE,
                    "top_p": self.config.TOP_P,
                    "num_predict": self.config.MAX_LLM_TOKENS,
                    "stop": ["Human:", "User:", "\n\n"]
                }
            }
            
            response = requests.post(
                self.config.OLLAMA_URL,
                json=payload,
                timeout=30  # 30 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return None
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            return None
    
    def build_context_prompt(self, current_text):
        context = ""
        if self.conversation_history:
            context = "Previous conversation:\n"
            for exchange in self.conversation_history[-2:]:
                context += f"Human: {exchange['human']}\nKrishna: {exchange['assistant']}\n"
            context += "\n"
        
        prompt = f"""You are Krishna, a helpful robot assistant. Keep responses very short (1 sentence, max 15 words).
Be natural, friendly, and conversational. Don't repeat the human's question.

{context}Human: {current_text}
Krishna:"""
        return prompt
    
    def handle_intent(self, text):
        text_lower = text.lower().strip()
        
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
        
        if any(word in text_lower for word in ["time", "what time"]):
            current_time = datetime.now().strftime("%I:%M %p")
            return f"It's {current_time}."
        
        if "date" in text_lower or "today" in text_lower:
            current_date = datetime.now().strftime("%A, %B %d")
            return f"Today is {current_date}."
        
        if any(word in text_lower for word in ["exit", "quit", "shutdown"]):
            return "EXIT"
        
        return self._get_llm_response(text)
    
    def _get_llm_response(self, text):
        try:
            prompt = self.build_context_prompt(text)
            response = self._call_ollama(prompt)
            
            if response:
                # Clean up the response
                response = response.strip()
                
                # Remove any potential system prompts or repetitions
                if "Krishna:" in response:
                    response = response.split("Krishna:")[-1].strip()
                
                # Limit response length
                words = response.split()
                if len(words) > self.config.MAX_RESPONSE_WORDS:
                    response = " ".join(words[:self.config.MAX_RESPONSE_WORDS])
                
                # Save to conversation history
                self.conversation_history.append({
                    "human": text,
                    "assistant": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                if len(self.conversation_history) > self.config.MAX_HISTORY:
                    self.conversation_history = self.conversation_history[-self.config.MAX_HISTORY:]
                
                return response
            else:
                return "Sorry, I had trouble processing that."
                
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, I had a processing error."
    
    # ---------------- SESSION ---------------- #
    def _save_session_data(self):
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
    
    # ---------------- WORKERS ---------------- #
    def process_audio_worker(self):
        while True:
            audio_file = self.audio_queue.get()
            if audio_file is None:
                break
            transcription = self.transcribe(audio_file)
            if transcription:
                print(f"📝 You said: '{transcription}'")
                response = self.handle_intent(transcription)
                if response:
                    self.text_queue.put(response)
            self.audio_queue.task_done()
    
    def process_text_worker(self):
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
    
    # ---------------- RUN ---------------- #
    def run(self):
        print("🤖 Krishna Robot Assistant v2.0 Online (Powered by Ollama)")
        print("Commands: [ENTER] Record | [s] Status | [t] Test TTS | [q] Quit")
        
        threading.Thread(target=self.process_audio_worker, daemon=True).start()
        threading.Thread(target=self.process_text_worker, daemon=True).start()
        
        if self.robot_awake:
            self.speak("Hello! I'm Krishna, your robot assistant. Ready to help!")

        try:
            while True:
                user_input = input("\n[ENTER] Record | [s] Status | [t] Test | [d] Debug | [q] Quit: ").strip().lower()
                
                if user_input == "q":
                    print("Shutting down Krishna...")
                    break
                elif user_input == "s":
                    status = "awake" if self.robot_awake else "sleeping"
                    print(f"Status: Krishna is {status}")
                elif user_input == "t":
                    self.speak("This is a test of the text to speech system.")
                elif user_input == "":
                    if self.record_audio_smart():
                        self.audio_queue.put(self.config.AUDIO_FILE)
                    else:
                        print("❌ Recording failed or too short. Please try again.")

        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received. Shutting down...")
        finally:
            self._save_session_data()
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
