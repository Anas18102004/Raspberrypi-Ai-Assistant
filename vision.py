import sounddevice as sd
import numpy as np
import requests
import asyncio
import os
import cv2
from PIL import Image
from scipy.io.wavfile import write
from deepgram import Deepgram
import google.generativeai as genai
from gtts import gTTS

# ---------------- CONFIGURATION ----------------
# Set up your API keys here.
# IMPORTANT: In a production environment, you should load these from environment variables
# for security instead of hardcoding them.
GEMINI_API_KEY = "AIzaSyDv1L2wgiR_FutCZFEeI_LcM15Ef0TUrY4"
DEEPGRAM_API_KEY = "ea93e67373ea77124ea2cb531678c691f289c714"
WEATHER_API_KEY = "53cea54c7ec545719c2151516250809"
CITY_NAME = "Ahmedabad,IN"  # Change to your desired city and country code

# Define the keywords that will trigger the weather or vision function.
WEATHER_KEYWORDS = ["weather", "whether", "forecast", "temperature", "climate", "rain", "sunny", "humidity", "wind"]
VISION_KEYWORDS = ["see", "look at", "describe", "what is", "what's", "show me", "analyze", "detect"]

# Audio recording settings.
DURATION = 5  # seconds of recording
FS = 44100    # sampling rate

# ---------------- INITIALIZATION ----------------
# Initialize the Gemini API client.
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ---------------- AUDIO FUNCTIONS ----------------
def record_audio():
    """Records audio from the microphone for a specified duration."""
    print("Listening...")
    # Record audio with the specified duration and sampling rate.
    recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype=np.int16)
    sd.wait()  # Wait until the recording is finished.
    filename = "voice.wav"
    write(filename, FS, recording)  # Save the recording to a WAV file.
    print("Recording complete.")
    return filename

def speak(text):
    """
    Converts text to speech using gTTS and plays the resulting audio file
    using the mpg123 command-line player.
    """
    print("Assistant:", text)
    tts = gTTS(text=text, lang='en')
    filename = "response.mp3"
    tts.save(filename)
    try:
        os.system(f"mpg123 {filename}")
    except Exception as e:
        print(f"Error playing sound with mpg123: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

# ---------------- VISION FUNCTION ----------------
def capture_frame():
    """Captures a single frame from the default webcam and returns the image as a NumPy array."""
    print("Capturing frame...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to capture image.")
        return None
    
    return frame

# ---------------- API FUNCTIONS ----------------
async def transcribe_audio(file_path):
    """Transcribes an audio file to text using the Deepgram API."""
    dg_client = Deepgram(DEEPGRAM_API_KEY)
    try:
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        source = {'buffer': audio_bytes, 'mimetype': 'audio/wav'}
        options = {'punctuate': True, 'language': 'en-US'}
        response = await dg_client.transcription.prerecorded(source, options)
        # Extract the transcript from the Deepgram response.
        transcript = response['results']['channels'][0]['alternatives'][0].get('transcript', '')
        print("You said:", transcript)
        return transcript
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

def get_weather():
    """Fetches and returns the current weather from the WeatherAPI.com."""
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={CITY_NAME}"
    try:
        res = requests.get(url)
        res.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        data = res.json()
        if data.get("current"):
            temp = data["current"]["temp_c"]
            desc = data["current"]["condition"]["text"]
            return f"The current temperature in {CITY_NAME} is {temp}°C with {desc}."
        else:
            return f"Sorry, couldn't get weather info. ({data.get('error', {}).get('message', 'No message')})"
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather: {e}"

def ask_gemini(question):
    """
    Asks a question to the Gemini model. If the question contains
    specific keywords, it calls the corresponding function (weather or vision).
    """
    try:
        # Check if the question is about weather.
        if any(word in question.lower() for word in WEATHER_KEYWORDS):
            return get_weather()
        
        # Check if the question is for a vision task.
        if any(word in question.lower() for word in VISION_KEYWORDS):
            frame = capture_frame()
            if frame is None:
                return "I couldn't access the camera to see."
            
            # Convert the OpenCV image (BGR) to a PIL image (RGB) in memory.
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            response = gemini_model.generate_content([question, image]).text
            return response
            
        # If no specific intent is detected, ask Gemini to answer generally.
        prompt = f"Please answer the following question in 1 or 2 short sentences:\n{question}"
        response = gemini_model.generate_content(prompt).text
        return response
    except Exception as e:
        print(f"Error while asking Gemini: {e}")
        return "Sorry, I couldn't process that."

# ---------------- MAIN EXECUTION ----------------
async def main():
    """The main loop for the voice assistant."""
    while True:
        audio_file = record_audio()
        text = await transcribe_audio(audio_file)

        # Remove the temporary audio file.
        os.remove(audio_file)

        if not text.strip():
            print("Didn't catch that. Please try again.")
            continue
        
        # Check for a specific "exit" command to terminate the script.
        if "exit" in text.lower() or "quit" in text.lower():
            speak("Goodbye!")
            break

        # Process the user's query and get a response.
        answer = ask_gemini(text)
        speak(answer)

if __name__ == "__main__":
    try:
        # Run the main asynchronous function.
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAssistant stopped.")
