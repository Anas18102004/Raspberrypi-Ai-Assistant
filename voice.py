import sounddevice as sd
import numpy as np
import requests
import json
import time
import asyncio
from scipy.io.wavfile import write
from deepgram import Deepgram
import google.generativeai as genai
from gtts import gTTS
import os

# ---------------- CONFIGURATION ----------------
GEMINI_API_KEY = "AIzaSyDv1L2wgiR_FutCZFEeI_LcM15Ef0TUrY4"
DEEPGRAM_API_KEY = "ea93e67373ea77124ea2cb531678c691f289c714"
OPENWEATHER_API_KEY = "1e613ff478e4a523cb3121caa909080a"
CITY_NAME = "Ahmedabad,IN"  # city, country code

# Initialize Gemini client
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel('gemini-1.5-flash')

# Define weather-related keywords
weather_keywords = ["weather","whether" ,"forecast", "temperature", "climate", "rain", "sunny", "humidity", "wind"]

# Audio settings
DURATION = 5  # seconds of recording
FS = 44100   # sampling rate

# ---------------- RECORD AUDIO ----------------
def record_audio():
    print("Listening...")
    recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype=np.int16)
    sd.wait()
    filename = "voice.wav"
    write(filename, FS, recording)
    print("Recording complete")
    return filename

# ---------------- DEEPGRAM STT ----------------
async def transcribe_audio(file_path):
    dg_client = Deepgram(DEEPGRAM_API_KEY)

    with open(file_path, 'rb') as f:
        audio_bytes = f.read()

    source = {'buffer': audio_bytes, 'mimetype': 'audio/wav'}
    options = {'punctuate': True, 'language': 'en-US'}

    response = await dg_client.transcription.prerecorded(source, options)
    transcript = response['results']['channels'][0]['alternatives'][0].get('transcript', '')
    print("You said:", transcript)
    return transcript

# ---------------- WEATHER FUNCTION ----------------
def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY_NAME}&appid={OPENWEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    data = res.json()
    if data.get("main"):
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"The current temperature in {CITY_NAME} is {temp}ï¿½C with {desc}."
    else:
        return "Sorry, I couldn't retrieve the weather information."

# ---------------- GEMINI Q&A ----------------
def ask_gemini(question):
    try:
        # Check if the question is about weather by matching any keyword
        if any(word in question.lower() for word in weather_keywords):
            return get_weather()
        
        # Ask Gemini to answer briefly
        prompt = f"Please answer the following question in 1 or 2 short sentences:\n{question}"
        response = gemini.generate_content(prompt).text
        return response
    except Exception as e:
        print("Error:", e)
        return "Sorry, I couldn't process that."

# ---------------- TEXT TO SPEECH ----------------
def speak(text):
    print("Assistant:", text)
    tts = gTTS(text=text, lang='en')
    filename = "response.mp3"
    tts.save(filename)
    os.system(f"mpg123 {filename}")  # Ensure mpg123 is installed or use another player

# ---------------- MAIN LOOP ----------------
async def main():
    while True:
        audio_file = record_audio()
        text = await transcribe_audio(audio_file)
        if text.strip() == "":
            print("Didn't catch that.")
            continue
        if "exit" in text.lower() or "quit" in text.lower():
            print("Goodbye!")
            break
        answer = ask_gemini(text)
        speak(answer)

if __name__ == "__main__":
    asyncio.run(main())
