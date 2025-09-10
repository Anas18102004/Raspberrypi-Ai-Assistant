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

# ---------------- CONFIGURATION ---------------- #
# Paths for models
WHISPER_MODEL_NAME = "base"  # Whisper model name
TINYLLAMA_MODEL_PATH = r"C:\Users\moham\tinyllama\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

AUDIO_FILE = "input.wav"
RECORD_SECONDS = 5
SAMPLE_RATE = 16000
MAX_LLM_TOKENS = 200
SILENCE_THRESHOLD = 0.01  # Threshold for detecting silence
MIN_RECORDING_LENGTH = 1  # Minimum recording length in seconds

# Wake words and sleep words
WAKE_WORDS = ["hey robot", "ok robot", "wake up", "hey krishna", "krishna"]
SLEEP_WORDS = ["sleep", "go to sleep", "stop listening", "goodbye"]

# Conversation history
CONVERSATION_HISTORY = []
MAX_HISTORY = 5  # Keep last 5 exchanges for context

# ---------------- LOGGING SETUP ---------------- #
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- INITIALIZE ---------------- #
# Text-to-Speech Engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
if voices:
    # Try to set a more natural voice if available
    engine.setProperty('voice', voices[0].id)
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Whisper Model
print("Loading Whisper model...")
try:
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    print("✅ Whisper loaded")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    exit(1)

# TinyLlama LLM
print("Loading TinyLlama model...")
try:
    llm = AutoModelForCausalLM.from_pretrained(
        TINYLLAMA_MODEL_PATH, 
        max_new_tokens=MAX_LLM_TOKENS,
        temperature=0.7,
        top_p=0.9
    )
    print("✅ TinyLlama loaded")
except Exception as e:
    logger.error(f"Failed to load TinyLlama model: {e}")
    exit(1)

# Queues for async processing
audio_queue = queue.Queue()
text_queue = queue.Queue()
last_response = ""
cache = {}
robot_awake = True  # Awake state

# ---------------- CORE FUNCTIONS ---------------- #
def speak(text):
    """Speak text and cache last response"""
    global last_response
    last_response = text
    cache["last_response"] = text
    cache["last_response_time"] = datetime.now().isoformat()
    
    print("🤖 Krishna:", text)
    logger.info(f"Speaking: {text}")
    
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"TTS error: {e}")

def detect_silence(audio_data, threshold=SILENCE_THRESHOLD):
    """Detect if audio contains mostly silence"""
    return np.mean(np.abs(audio_data)) < threshold

def record_audio_smart(max_duration=RECORD_SECONDS):
    """Record audio with smart silence detection"""
    print("🎙️ Recording... speak now!")
    
    try:
        # Start recording
        recording = []
        chunk_duration = 0.1  # 100ms chunks
        chunk_size = int(chunk_duration * SAMPLE_RATE)
        silence_duration = 0
        max_silence = 2.0  # Stop after 2 seconds of silence
        
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
        stream.start()
        
        start_time = time.time()
        has_speech = False
        
        while time.time() - start_time < max_duration:
            chunk, _ = stream.read(chunk_size)
            chunk = chunk.flatten()
            recording.extend(chunk)
            
            # Check for silence
            if detect_silence(chunk):
                silence_duration += chunk_duration
            else:
                silence_duration = 0
                has_speech = True
            
            # Stop if we've had speech and then silence
            if has_speech and silence_duration >= max_silence:
                break
            
            # Show recording time
            elapsed = time.time() - start_time
            print(f"Recording... {elapsed:.1f}s", end="\r")
        
        stream.stop()
        stream.close()
        
        # Convert to numpy array and save
        recording = np.array(recording, dtype=np.float32)
        
        # Convert to int16 for saving
        recording_int16 = (recording * 32767).astype(np.int16)
        
        with wave.open(AUDIO_FILE, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(recording_int16.tobytes())
        
        print(f"\nRecording saved ({time.time() - start_time:.1f}s)")
        return len(recording) / SAMPLE_RATE >= MIN_RECORDING_LENGTH
        
    except Exception as e:
        logger.error(f"Recording error: {e}")
        return False

def transcribe(audio_file):
    """Transcribe audio using Whisper with error handling"""
    try:
        result = whisper_model.transcribe(audio_file, fp16=False)
        text = result["text"].strip()
        
        # Log confidence if available
        if "segments" in result and result["segments"]:
            avg_confidence = np.mean([seg.get("no_speech_prob", 0) for seg in result["segments"]])
            logger.info(f"Transcription confidence: {1-avg_confidence:.2f}")
        
        return text
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

def build_context_prompt(current_text):
    """Build prompt with conversation history for better context"""
    context = ""
    if CONVERSATION_HISTORY:
        context = "Previous conversation:\n"
        for exchange in CONVERSATION_HISTORY[-3:]:  # Last 3 exchanges
            context += f"Human: {exchange['human']}\nKrishna: {exchange['assistant']}\n"
        context += "\n"
    
    prompt = f"""You are Krishna, a helpful and friendly robot assistant. You speak naturally and conversationally.
Keep your responses concise (1-2 sentences max) and helpful.

{context}Human: {current_text}
Krishna:"""
    
    return prompt

def handle_intent(text):
    """Handle commands and ask LLM with improved context"""
    global robot_awake
    text_lower = text.lower().strip()
    
    # Ignore very short or empty transcriptions
    if len(text.strip()) < 2:
        return None

    # Sleep/Wake control
    if any(word in text_lower for word in SLEEP_WORDS):
        robot_awake = False
        return "Going to sleep. Say 'hey Krishna' to wake me up."
    
    if any(word in text_lower for word in WAKE_WORDS):
        if not robot_awake:
            robot_awake = True
            return "Hello! I'm Krishna, your robot assistant. How can I help you?"
        else:
            return "I'm already awake! What can I do for you?"

    if not robot_awake:
        return None  # Ignore commands while sleeping

    # Built-in commands with better responses
    if any(word in text_lower for word in ["time", "what time"]):
        current_time = datetime.now().strftime("%I:%M %p")
        return f"It's {current_time}."
    
    if "date" in text_lower or "today" in text_lower:
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {current_date}."
    
    if "weather" in text_lower:
        return "I can't check live weather data, but I'd recommend checking your weather app."
    
    if "repeat" in text_lower or "say that again" in text_lower:
        return cache.get("last_response", "I don't have anything to repeat.")
    
    if any(word in text_lower for word in ["exit", "quit", "shutdown"]):
        return "EXIT"
    
    if "clear history" in text_lower:
        CONVERSATION_HISTORY.clear()
        return "I've cleared our conversation history."

    # Ask TinyLlama with context
    try:
        prompt = build_context_prompt(text)
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        response = llm(prompt)
        
        if isinstance(response, str):
            # Very aggressive cleaning
            response = response.strip()
            
            # Remove common prefixes immediately
            prefixes_to_remove = ["Krishna:", "Assistant:", "AI:", "Robot:", "Human:", "No, no, no."]
            for prefix in prefixes_to_remove:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            
            # Take only first sentence and limit to 10 words max
            sentences = response.replace('!', '.').replace('?', '.').split('.')
            response = sentences[0].strip()
            
            words = response.split()
            if len(words) > 10:
                response = ' '.join(words[:10])
            
            # Add punctuation if missing
            if response and response[-1] not in '.!?':
                response += '.'
            
            # Don't add to history to avoid token buildup
            logger.info(f"Final response: {response}")
            return response if len(response) > 2 else "I don't understand."
            
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "Sorry, I had an error."

    return "I don't understand."

def save_session_data():
    """Save conversation history and cache"""
    try:
        session_data = {
            "conversation_history": CONVERSATION_HISTORY,
            "cache": cache,
            "timestamp": datetime.now().isoformat()
        }
        with open("session_data.json", "w") as f:
            json.dump(session_data, f, indent=2)
        logger.info("Session data saved")
    except Exception as e:
        logger.error(f"Failed to save session data: {e}")

def load_session_data():
    """Load previous conversation history and cache"""
    global CONVERSATION_HISTORY, cache
    try:
        if os.path.exists("session_data.json"):
            with open("session_data.json", "r") as f:
                session_data = json.load(f)
            CONVERSATION_HISTORY = session_data.get("conversation_history", [])
            cache.update(session_data.get("cache", {}))
            logger.info("Session data loaded")
    except Exception as e:
        logger.error(f"Failed to load session data: {e}")

# ---------------- THREAD FUNCTIONS ---------------- #
def process_audio():
    """Process audio transcription in background"""
    while True:
        audio_file = audio_queue.get()
        if audio_file is None:
            break
        
        transcription = transcribe(audio_file)
        if transcription and len(transcription.strip()) > 1:
            print(f"🗣️ You said: {transcription}")
            logger.info(f"Transcribed: {transcription}")
            
            response = handle_intent(transcription)
            if response:
                text_queue.put(response)
        else:
            logger.info("No clear speech detected")
            
        audio_queue.task_done()

def process_text():
    """Process text-to-speech in background"""
    while True:
        text = text_queue.get()
        if text is None:
            break
        if text == "EXIT":
            print("👋 Krishna is shutting down...")
            save_session_data()
            os._exit(0)
        speak(text)
        text_queue.task_done()

# ---------------- MAIN LOOP ---------------- #
def main():
    """Main application loop"""
    print("🤖 Krishna Robot Assistant Online")
    print("Press ENTER to record, 's' for status, 'q' to quit.")
    
    # Load previous session
    load_session_data()
    
    # Start background threads
    threading.Thread(target=process_audio, daemon=True).start()
    threading.Thread(target=process_text, daemon=True).start()
    
    # Initial greeting
    if robot_awake:
        speak("Hello! I'm Krishna, your robot assistant. How can I help you today?")

    try:
        while True:
            user_input = input("\n[ENTER] Record | [s] Status | [q] Quit: ").strip().lower()
            
            if user_input == "q":
                print("Shutting down Krishna...")
                save_session_data()
                break
            elif user_input == "s":
                status = "awake" if robot_awake else "sleeping"
                print(f"Status: Krishna is {status}")
                print(f"Conversation history: {len(CONVERSATION_HISTORY)} exchanges")
                continue
            
            # Record audio
            if record_audio_smart():
                audio_queue.put(AUDIO_FILE)
            else:
                print("Recording too short or failed, please try again.")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down...")
        save_session_data()
    
    # Cleanup
    audio_queue.put(None)
    text_queue.put(None)

if __name__ == "__main__":
    main()
