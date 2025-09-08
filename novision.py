import cv2
import base64
import os
import google.genai as genai
from deepgram import DeepgramClient, PrerecordedOptions
import asyncio
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

# Load API Keys
GEMINI_API_KEY = "AIzaSyDv1L2wgiR_FutCZFEeI_LcM15Ef0TUrY4"
DEEPGRAM_API_KEY = "ea93e67373ea77124ea2cb531678c691f289c714"

client = genai.Client(api_key=GEMINI_API_KEY)
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

# Global variables
cap = None
latest_frame = None
is_listening = False
httpd = None
server_thread = None
server_port = None

# -----------------------------
# CAMERA FUNCTIONS
# -----------------------------
def initialize_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Webcam not accessible")
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

def camera_loop():
    """Continuously capture frames and display camera feed"""
    global latest_frame, cap
    
    while True:
        if cap is None:
            break
            
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        latest_frame = frame.copy()
        
        # Add status text overlay
        status_text = "🎙 LISTENING..." if is_listening else "💬 Say something to ask about what you see"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Voice-Activated Camera Assistant', frame)
        
        # Check for 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cleanup_camera()

def save_current_frame():
    """Save the current frame as an image"""
    global latest_frame
    if latest_frame is not None:
        img_path = "captured_frame.jpg"
        cv2.imwrite(img_path, latest_frame)
        return img_path
    return None

def cleanup_camera():
    global cap
    if cap:
        cap.release()
    cv2.destroyAllWindows()

def start_static_server(directory: str = "frontend", preferred_port: int = 8000) -> int:
    """Start a background HTTP server serving the given directory. Returns the chosen port."""
    global httpd, server_thread, server_port

    class Handler(http.server.SimpleHTTPRequestHandler):
        def _init_(self, *args, **kwargs):
            # Serve files from the specified directory
            super()._init_(*args, directory=directory, **kwargs)

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
# GEMINI - TEXT ONLY (NO VISION)
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
# SPEECH TO TEXT (STT) - Fixed for Deepgram v4.0.0+
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
        # Convert numpy array to WAV format properly
        temp_wav = "temp_audio.wav"
        
        # Write proper WAV file
        with wave.open(temp_wav, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(fs)  # sample rate
            wav_file.writeframes(audio.tobytes())
        
        # Read the WAV file as bytes
        with open(temp_wav, 'rb') as f:
            audio_data = f.read()
        
        # Configure options for Deepgram v4+
        options = PrerecordedOptions(
            model="nova-2",
            punctuate=True,
            language="en-US"
        )
        
        # Use the correct v4+ API (no await needed)
        response = deepgram.listen.rest.v("1").transcribe_file(
            {"buffer": audio_data},
            options
        )
        
        # Extract transcript from response structure
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        print("📝 You said:", transcript)
        
        # Clean up temp file
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        return transcript
        
    except Exception as e:
        print(f"Error in transcription: {e}")
        # Clean up temp file on error
        if os.path.exists("temp_audio.wav"):
            try:
                os.remove("temp_audio.wav")
            except:
                pass
        return ""

# -----------------------------
# TEXT TO SPEECH (TTS) - Improved Windows Compatible
# -----------------------------
def speak(text):
    print(f"🔊 Speaking: {text}")  # Always show what we're trying to say
    
    try:
        # Method 1: Windows PowerShell TTS (most reliable on Windows)
        try:
            import subprocess
            # Clean the text for PowerShell
            cleaned_text = text.replace('"', "'").replace("'", "''")
            # Use a more robust PowerShell command
            cmd = [
                'powershell', '-Command',
                f"Add-Type -AssemblyName System.Speech; "
                f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$synth.Speak('{cleaned_text}'); "
                f"$synth.Dispose()"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print("✅ TTS completed successfully")
                return
            else:
                print(f"PowerShell TTS failed: {result.stderr}")
        except Exception as e:
            print(f"PowerShell TTS error: {e}")
        
        # Method 2: Try pyttsx3 (offline TTS)
        try:
            import pyttsx3
            # Create fresh engine each time to avoid conflicts
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Speed of speech
            engine.setProperty('volume', 0.9)  # Volume level
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine  # Clean up
            print("✅ pyttsx3 TTS completed")
            return
        except Exception as e:
            print(f"pyttsx3 TTS error: {e}")
        
        # Method 3: Try gTTS with Windows Media Player
        try:
            tts = gTTS(text=text, lang="en")
            temp_file = f"temp_audio_{int(time.time())}.mp3"  # Unique filename
            tts.save(temp_file)
            
            # Use Windows Media Player
            import subprocess
            subprocess.run([
                'powershell', '-Command',
                f'(New-Object -com wmplayer.ocx.7).controls.play(); '
                f'Start-Process -FilePath "{os.path.abspath(temp_file)}" -Wait'
            ], capture_output=True, timeout=15)
            
            # Clean up
            time.sleep(1)  # Give time for file to finish playing
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass  # File might still be in use
            
            print("✅ gTTS + Media Player completed")
            return
            
        except Exception as e:
            print(f"gTTS error: {e}")
        
        # Method 4: Simple Windows 'say' command alternative
        try:
            import subprocess
            # Use Windows built-in narrator
            subprocess.run(['narrator', '/speech', text], capture_output=True, timeout=10)
            print("✅ Windows Narrator completed")
            return
        except Exception as e:
            print(f"Windows Narrator error: {e}")
        
        print("❌ All TTS methods failed - check audio system")
        
    except Exception as e:
        print(f"❌ Critical TTS error: {e}")
        print(f"📝 Text was: {text}")

# -----------------------------
# VOICE COMMAND PROCESSING
# -----------------------------
def process_voice_command():
    """Record audio, transcribe, and get AI response (text-only)."""
    try:
        # Step 1: Record and transcribe user question
        user_question = record_and_transcribe()

        if not user_question or len(user_question.strip()) < 3:
            print("❌ No clear question detected. Try again.")
            return

        print(f"🗣 You asked: '{user_question}'")

        # Step 2: Send question text to Gemini (no image)
        print("🤖 Processing with Gemini (text-only)...")
        answer = ask_gemini_text(user_question)
        print(f"🤖 Gemini: {answer}")

        # Step 3: Speak the response
        speak(answer)

    except Exception as e:
        print(f"Error processing voice command: {e}")

# -----------------------------
# CONTINUOUS VOICE MONITORING
# -----------------------------
def voice_monitoring_loop():
    """Continuously monitor for voice commands (no camera dependency)."""
    print("\n🎯 Voice monitoring started!")
    print("💡 Tips:")
    print("   - Ask questions like: 'What's the weather?', 'Tell me a joke', 'Explain quantum computing' ")
    print("   - Speak clearly for 5 seconds when the microphone activates\n")

    while True:
        try:
            # Wait a moment between interactions
            time.sleep(2)

            # Process voice command
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
        # Start local static server for reliable autoplay and open browser
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
        cleanup_camera()  # Safe to call even if camera wasn't used
        # Stop the HTTP server if running
        try:
            if httpd is not None:
                httpd.shutdown()
                httpd.server_close()
        except Exception:
            pass
        print("👋 Program ended")

if _name_ == "_main_":
    main()
