import sounddevice as sd
import numpy as np
import requests
import asyncio
import os
import json
import websockets
import google.generativeai as genai
from elevenlabs import play, stream, Voice, VoiceSettings
import RPi.GPIO as GPIO
import time
from RPLCD.i2c import CharLCD

# ---------------- CONFIGURATION ----------------
# Use environment variables for secure API key access
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

CITY_NAME = "Ahmedabad,IN"
WEATHER_KEYWORDS = ["weather", "forecast", "temperature", "climate", "rain", "sunny", "humidity", "wind"]

FS = 44100
CHANNELS = 1
BLOCK_SIZE = 1024  # Smaller blocks for lower latency

# ---------------- GPIO & LCD SETUP ----------------
LED_BLUE = 21
LED_WHITE = 20
LED_R5 = 25
LED_E = 24
LED_D4 = 23
LED_D5 = 17
LED_D6 = 18
LED_D7 = 22

ALL_LEDS = [LED_BLUE, LED_WHITE, LED_R5, LED_E, LED_D4, LED_D5, LED_D6, LED_D7]

GPIO.setmode(GPIO.BCM)
for led in ALL_LEDS:
    GPIO.setup(led, GPIO.OUT)
    GPIO.output(led, GPIO.LOW)

lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8,
              charmap='A02', auto_linebreaks=True)

# ---------------- INITIALIZATION ----------------
if not all([GEMINI_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY, OPENWEATHER_API_KEY]):
    raise ValueError("One or more API keys are not set as environment variables.")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')
ELEVENLABS_VOICE_ID = "YOUR_ELEVENLABS_VOICE_ID" # You can find this in your ElevenLabs account

# ---------------- UTILITY FUNCTIONS ----------------
def turn_on(led):
    GPIO.output(led, GPIO.HIGH)

def turn_off(led):
    GPIO.output(led, GPIO.LOW)

def all_leds_off():
    for led in ALL_LEDS:
        GPIO.output(led, GPIO.LOW)

def indicate_listening():
    all_leds_off()
    turn_on(LED_BLUE)

def indicate_processing():
    all_leds_off()
    turn_on(LED_WHITE)

def indicate_error():
    all_leds_off()
    turn_on(LED_R5)
    time.sleep(2)
    turn_off(LED_R5)

def display_weather(temp, condition):
    lcd.clear()
    lcd.write_string("Weather:")
    lcd.crlf()
    message = f"{temp:.1f}C, {condition}"
    lcd.write_string(message[:16])

def clear_display():
    lcd.clear()

def display_weather_on_leds(temp):
    all_leds_off()
    temp_ranges = [10, 15, 20, 25, 30, 35, 40, 45]
    for i, threshold in enumerate(temp_ranges):
        if temp >= threshold:
            turn_on(ALL_LEDS[i])
        else:
            break

# ---------------- API FUNCTIONS ----------------
def speak(text):
    """Uses ElevenLabs streaming API for low-latency TTS."""
    print("Assistant:", text)
    try:
        audio_stream = play(
            text=text,
            voice=Voice(voice_id=ELEVENLABS_VOICE_ID, settings=VoiceSettings(stability=0.5, similarity_boost=0.75)),
            stream=True
        )
        for chunk in audio_stream:
            # The play function handles streaming playback automatically
            pass 
        
    except Exception as e:
        print(f"Error during ElevenLabs TTS: {e}")
        indicate_error()

def get_weather_data(city_name, api_key):
    """Fetches real-time weather data from OpenWeatherMap."""
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
    try:
        response = requests.get(complete_url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data["cod"] == 200:
            temp = data["main"]["temp"]
            condition = data["weather"][0]["description"]
            return {"temperature_celsius": temp, "condition": condition}
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def ask_gemini(question):
    """Processes user question and generates a response."""
    try:
        if any(word in question.lower() for word in WEATHER_KEYWORDS):
            weather_info = get_weather_data(CITY_NAME, OPENWEATHER_API_KEY)
            if weather_info:
                temp = weather_info['temperature_celsius']
                condition = weather_info['condition']
                
                # Gemini generates a conversational response based on real data
                prompt = f"Given the weather data: temperature is {temp:.1f}°C and condition is {condition}, generate a short, one-sentence spoken response about the weather."
                response_text = gemini_model.generate_content(prompt).text
                
                display_weather(temp, condition)
                display_weather_on_leds(float(temp))
                return response_text
            else:
                return "Sorry, I couldn't get the current weather information."
        
        # General question handling
        prompt = f"Please answer the following question in 1 or 2 short sentences:\n{question}"
        response = gemini_model.generate_content(prompt).text
        return response
    
    except Exception as e:
        indicate_error()
        print(f"Error while asking Gemini: {e}")
        return "Sorry, I couldn't process that."

async def transcribe_stream():
    """Streams audio to Deepgram in real-time for low-latency transcription."""
    deepgram_url = "wss://api.deepgram.com/v1/listen?punctuate=true&model=nova-2-general"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

    try:
        indicate_listening()
        async with websockets.connect(deepgram_url, extra_headers=headers) as ws:
            transcription_complete = asyncio.Event()
            transcribed_text = ""

            async def receive_transcripts():
                nonlocal transcribed_text
                async for message in ws:
                    response = json.loads(message)
                    if response.get('is_final'):
                        transcript = response['channel']['alternatives'][0]['transcript']
                        transcribed_text = transcript
                        print(f"You said: {transcript}")
                        transcription_complete.set()
                        break
            
            # Start the listener task
            listener_task = asyncio.create_task(receive_transcripts())
            
            with sd.InputStream(samplerate=FS, channels=CHANNELS, dtype=np.int16, blocksize=BLOCK_SIZE) as mic_stream:
                print("Listening for a complete sentence...")
                while not transcription_complete.is_set():
                    data, overflowed = mic_stream.read(mic_stream.read.blocksize, False)
                    await ws.send(data.tobytes())
                
                await ws.close()
                return transcribed_text
    
    except Exception as e:
        indicate_error()
        print(f"Error during transcription streaming: {e}")
        return ""

# ---------------- MAIN LOOP ----------------
async def main():
    print("Voice Assistant Initialized. Press Ctrl+C to stop.")
    clear_display()
    lcd.write_string("Assistant Ready")
    time.sleep(2)
    clear_display()
    all_leds_off()

    while True:
        text = await transcribe_stream()
        all_leds_off()
        
        if not text.strip():
            print("No speech detected.")
            speak("I didn't hear you. Please try again.")
            continue
        
        if "exit" in text.lower() or "quit" in text.lower():
            speak("Goodbye!")
            clear_display()
            lcd.write_string("Goodbye!")
            time.sleep(2)
            break
        
        indicate_processing()
        answer = ask_gemini(text)
        all_leds_off()
        speak(answer)
        all_leds_off()
        clear_display()
        
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAssistant stopped.")
    finally:
        all_leds_off()
        clear_display()
        GPIO.cleanup()
