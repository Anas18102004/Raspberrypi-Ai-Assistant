import os
import google.genai as genai
from deepgram import DeepgramClient, PrerecordedOptions
import sounddevice as sd
import numpy as np
from gtts import gTTS
import tempfile
import threading
import time
import wave
import webbrowser
from pathlib import Path
import http.server
import socketserver
import socket
import subprocess

# Load API Keys
# NOTE: The API keys are provided in the environment and should not be modified.
GEMINI_API_KEY = "AIzaSyDv1L2wgiR_FutCZFEeI_LcM15Ef0TUrY4"
DEEPGRAM_API_KEY = "ea93e67373ea77124ea2cb531678c691f289c714"

# Initialize clients
client = genai.Client(api_key=GEMINI_API_KEY)
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

# Global variables
is_listening = False
httpd = None
server_thread = None
server_port = None

# -----------------------------
# STATIC WEB SERVER (Optional for a simple UI)
# -----------------------------
def start_static_server(directory: str = "frontend", preferred_port: int = 8000) -> int:
    """Start a background HTTP server serving the given directory. Returns the chosen port."""
    global httpd, server_thread, server_port

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            # Serve files from the specified directory
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, format, *args):
            # Quieter logs
            pass

    # Try preferred port; if busy, let OS choose a free one
    addr = ("127.0.0.1", preferred_port)
    try:
        httpd = socketserver.ThreadingTCPServer(addr, Handler)
    except OSError:
        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)

    server_port = httpd.server_address[1]
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    return server_port

# -----------------------------
# GEMINI - TEXT ONLY
# -----------------------------
def ask_gemini_text(prompt: str) -> str:
    """Send plain text to Gemini and return the response text."""
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )

        if hasattr(response, 'text') and response.text:
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            cand = response.candidates[0]
            if hasattr(cand, 'content') and cand.content and cand.content.parts:
                return cand.content.parts[0].text
            if hasattr(cand, 'text') and cand.text:
                return cand.text
        # Fallback
        return "I'm not sure how to respond to that."
    except Exception as e:
        print(f"Error calling Gemini (text): {e}")
        return "There was an error contacting the AI service."

# -----------------------------
# SPEECH TO TEXT (STT) - Deepgram
# -----------------------------
def record_and_transcribe():
    global is_listening
    is_listening = True
    
    print("\n🎙 Listening for 5 seconds...")
    duration = 5  # seconds
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    
    is_listening = False
    
    try:
        temp_wav = "temp_audio.wav"
        
        with wave.open(temp_wav, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(fs)  # sample rate
            wav_file.writeframes(audio.tobytes())
        
        with open(temp_wav, 'rb') as f:
            audio_data = f.read()
        
        options = PrerecordedOptions(
            model="nova-2",
            punctuate=True,
            language="en-US"
        )
        
        response = deepgram.listen.rest.v("1").transcribe_file(
            {"buffer": audio_data},
            options
        )
        
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        print("📝 You said:", transcript)
        
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        return transcript
            
    except Exception as e:
        print(f"Error in transcription: {e}")
        if os.path.exists("temp_audio.wav"):
            try:
                os.remove("temp_audio.wav")
            except:
                pass
        return ""

# -----------------------------
# TEXT TO SPEECH (TTS) - Cross-platform
# -----------------------------
def speak(text):
    print(f"🔊 Speaking: {text}")
    try:
        # Use gTTS for a more universal solution
        tts = gTTS(text=text, lang="en", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            temp_file_path = fp.name
            tts.save(temp_file_path)

        # Play the audio file using a cross-platform command
        # This will work on most Linux systems (including Raspberry Pi)
        # and macOS. On Windows, it will try to open with a default player.
        try:
            subprocess.run(["mpg123", temp_file_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            # Fallback for systems without mpg123
            try:
                subprocess.run(["ffplay", "-nodisp", "-autoexit", temp_file_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                print("Warning: mpg123 or ffplay not found. Cannot play audio.")
                
        # Clean up the temporary file
        os.remove(temp_file_path)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

# -----------------------------
# VOICE COMMAND PROCESSING
# -----------------------------
def process_voice_command():
    """Record audio, transcribe, and get AI response (text-only)."""
    try:
        user_question = record_and_transcribe()

        if not user_question or len(user_question.strip()) < 3:
            print("❌ No clear question detected. Try again.")
            return

        print(f"🗣 You asked: '{user_question}'")
        print("🤖 Processing with Gemini (text-only)...")
        answer = ask_gemini_text(user_question)
        print(f"🤖 Gemini: {answer}")
        speak(answer)

    except Exception as e:
        print(f"Error processing voice command: {e}")

# -----------------------------
# CONTINUOUS VOICE MONITORING
# -----------------------------
def voice_monitoring_loop():
    """Continuously monitor for voice commands."""
    print("\n🎯 Voice monitoring started!")
    print("💡 Tips:")
    print("    - Ask questions like: 'What's the weather?', 'Tell me a joke', 'Explain quantum computing' ")
    print("    - Speak clearly for 5 seconds when the microphone activates\n")

    while True:
        try:
            time.sleep(2)
            process_voice_command()

        except KeyboardInterrupt:
            print("\n🛑 Voice monitoring stopped by user")
            break
        except Exception as e:
            print(f"Error in voice monitoring: {e}")
            time.sleep(1)

# -----------------------------
# MAIN
# -----------------------------
def main():
    try:
        print("🚀 Starting Voice-Activated Assistant (Text-Only Mode)...")
        
        # Start local static server for reliable autoplay and open browser (optional)
        try:
            port = start_static_server(directory="frontend", preferred_port=8000)
            url = f"http://127.0.0.1:{port}/index.html"
            webbrowser.open(url, new=1)
            print(f"🌐 Opened browser at: {url}")
        except Exception as e:
            print(f"⚠ Could not start static server or open browser automatically: {e}")

        # Start voice monitoring (this will run in main thread)
        voice_monitoring_loop()

    except KeyboardInterrupt:
        print("\n🛑 Program stopped by user")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Stop the HTTP server if running
        try:
            if httpd is not None:
                httpd.shutdown()
                httpd.server_close()
        except Exception:
            pass
        print("👋 Program ended")

if __name__ == "__main__":
    main()
