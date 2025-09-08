import os
import time
import google.generativeai as genai
from deepgram import DeepgramClient, PrerecordedOptions
from gtts import gTTS
from dotenv import load_dotenv
from playsound import playsound
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
RECORDING_DURATION = 5  # seconds
SAMPLE_RATE = 16000     # Hertz
TEMP_AUDIO_FILENAME = "temp_audio.wav"
TEMP_RESPONSE_FILENAME = "temp_response.mp3"

# --- Initialization ---
try:
    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # Configure Deepgram
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
except Exception as e:
    print(f"Error during initialization: {e}")
    exit()

def record_audio(duration, fs, filename):
    """Records audio from the microphone and saves it as a WAV file."""
    print("🎙️  Listening...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, recording)  # Save as WAV file
    print("✅ Finished listening.")
    return filename

def transcribe_audio(audio_path):
    """Transcribes audio using Deepgram."""
    try:
        with open(audio_path, 'rb') as audio_file:
            source = {'buffer': audio_file, 'mimetype': 'audio/wav'}
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                punctuate=True,
            )
            print("🧠 Transcribing with Deepgram...")
            response = deepgram.listen.prerecorded.v("1").transcribe_file(source, options)
            transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
            print("✅ Transcription complete.")
            return transcript
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

def get_gemini_response(prompt):
    """Gets a response from the Gemini model."""
    try:
        print("🤖 Thinking with Gemini...")
        response = gemini_model.generate_content(prompt)
        print("✅ Got response from Gemini.")
        return response.text
    except Exception as e:
        print(f"Error getting response from Gemini: {e}")
        return "Sorry, I couldn't process that request."

def speak_text(text):
    """Converts text to speech using gTTS and plays it."""
    try:
        print("🔊 Speaking...")
        tts = gTTS(text=text, lang='en')
        tts.save(TEMP_RESPONSE_FILENAME)
        playsound(TEMP_RESPONSE_FILENAME)
        os.remove(TEMP_RESPONSE_FILENAME)
        print("✅ Finished speaking.")
    except Exception as e:
        print(f"Error during text-to-speech: {e}")

def main():
    """Main loop for the AI assistant."""
    print("👋 Hello! I'm your AI assistant. Say 'goodbye' to exit.")
    
    while True:
        # 1. Listen for user's voice
        audio_file = record_audio(RECORDING_DURATION, SAMPLE_RATE, TEMP_AUDIO_FILENAME)
        
        # 2. Transcribe the audio to text
        user_prompt = transcribe_audio(audio_file)
        
        # Clean up the temporary audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)
            
        if not user_prompt:
            print("Could not hear anything, please try again.")
            continue
            
        print(f"👤 You said: {user_prompt}")
        
        # Check for exit command
        if "goodbye" in user_prompt.lower():
            speak_text("Goodbye!")
            break
            
        # 3. Get a response from Gemini
        ai_response = get_gemini_response(user_prompt)
        print(f"💡 Gemini: {ai_response}")
        
        # 4. Speak the response
        speak_text(ai_response)
        
        # Wait a moment before listening again
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Assistant stopped by user.")
    finally:
        # Final cleanup of temp files if they exist
        if os.path.exists(TEMP_AUDIO_FILENAME):
            os.remove(TEMP_AUDIO_FILENAME)
        if os.path.exists(TEMP_RESPONSE_FILENAME):
            os.remove(TEMP_RESPONSE_FILENAME)
        print("✨ Program ended.")
