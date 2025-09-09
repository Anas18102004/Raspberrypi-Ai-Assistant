import sounddevice as sd
import numpy as np
import requests
import asyncio
import os
import json
from scipy.io.wavfile import write
from deepgram import Deepgram
import google.generativeai as genai
from gtts import gTTS
import RPi.GPIO as GPIO
import time
from RPLCD.i2c import CharLCD

# ---------------- CONFIGURATION ----------------
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
DEEPGRAM_API_KEY = "YOUR_DEEPGRAM_API_KEY"
WEATHER_API_KEY = "YOUR_WEATHER_API_KEY" # Not used for getting weather data directly, but can be for other weather queries.
CITY_NAME = "Ahmedabad,IN"

WEATHER_KEYWORDS = ["weather", "forecast", "temperature", "climate", "rain", "sunny", "humidity", "wind"]

DURATION = 5
FS = 44100

# ---------------- GPIO SETUP ----------------
LED_BLUE = 21   # GPIO21, Pin 40
LED_WHITE = 20  # GPIO20, Pin 38
LED_R5 = 25     # GPIO25, Pin 22
LED_E = 24      # GPIO24, Pin 18
LED_D4 = 23     # GPIO23, Pin 16
LED_D5 = 17     # GPIO17, Pin 11
LED_D6 = 18     # GPIO18, Pin 12
LED_D7 = 22     # GPIO22, Pin 15

ALL_LEDS = [LED_BLUE, LED_WHITE, LED_R5, LED_E, LED_D4, LED_D5, LED_D6, LED_D7]

GPIO.setmode(GPIO.BCM)
for led in ALL_LEDS:
    GPIO.setup(led, GPIO.OUT)
    GPIO.output(led, GPIO.LOW)

# ---------------- LCD SETUP ----------------
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8,
              charmap='A02', auto_linebreaks=True)

# ---------------- INITIALIZATION ----------------
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

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
    lcd.write_string("Weather Info:")
    lcd.crlf()
    message = f"{temp}C, {condition}"
    lcd.write_string(message[:16])

def clear_display():
    lcd.clear()

def display_weather_on_leds(temp):
    all_leds_off()

    # Define temperature ranges for each LED
    temp_ranges = [10, 15, 20, 25, 30, 35, 40, 45]

    for i, threshold in enumerate(temp_ranges):
        if temp >= threshold:
            turn_on(ALL_LEDS[i])
        else:
            break

# ---------------- AUDIO FUNCTIONS ----------------
def record_audio():
    print("Listening...")
    indicate_listening()
    recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype=np.int16)
    sd.wait()
    all_leds_off()
    filename = "voice.wav"
    write(filename, FS, recording)
    print("Recording complete.")
    return filename

def speak(text):
    print("Assistant:", text)
    tts = gTTS(text=text, lang='en')
    filename = "response.mp3"
    tts.save(filename)
    try:
        os.system(f"mpg123 {filename}")
    except Exception as e:
        print(f"Error playing sound: {e}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

# ---------------- API FUNCTIONS ----------------
async def transcribe_audio(file_path):
    dg_client = Deepgram(DEEPGRAM_API_KEY)
    try:
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        source = {'buffer': audio_bytes, 'mimetype': 'audio/wav'}
        options = {'punctuate': True, 'language': 'en-US'}
        indicate_processing()
        response = await dg_client.transcription.prerecorded(source, options)
        all_leds_off()
        transcript = response['results']['channels'][0]['alternatives'][0].get('transcript', '')
        print("You said:", transcript)
        return transcript
    except Exception as e:
        indicate_error()
        print(f"Error during transcription: {e}")
        return ""

def ask_gemini(question):
    try:
        # Check for weather-related keywords
        if any(word in question.lower() for word in WEATHER_KEYWORDS):
            # Tell Gemini to provide a JSON response with weather details
            prompt = f"Using real-time data, what is the current weather in {CITY_NAME}? Respond in a single JSON object with 'temperature_celsius', 'condition', and 'spoken_response' keys. The 'spoken_response' should be a short sentence."

            response = gemini_model.generate_content(prompt).text

            # Parse the JSON response
            try:
                weather_data = json.loads(response)
                temp = weather_data.get('temperature_celsius')
                condition = weather_data.get('condition')
                spoken_response = weather_data.get('spoken_response')

                if temp is not None and condition:
                    # Display data on LCD and LEDs
                    display_weather(temp, condition)
                    display_weather_on_leds(float(temp))

                    # Return the spoken response for text-to-speech
                    return spoken_response
                else:
                    return "Sorry, I couldn't get the specific weather data from Gemini."

            except json.JSONDecodeError:
                # If Gemini doesn't return valid JSON, just speak its response as is
                print("Gemini response was not valid JSON.")
                return response

        # For non-weather questions, use the original prompt
        prompt = f"Please answer the following question in 1 or 2 short sentences:\n{question}"
        response = gemini_model.generate_content(prompt).text
        return response

    except Exception as e:
        indicate_error()
        print(f"Error while asking Gemini: {e}")
        return "Sorry, I couldn't process that."

# ---------------- MAIN LOOP ----------------
async def main():
    print("Voice Assistant Initialized. Press Ctrl+C to stop.")
    clear_display()
    lcd.write_string("Assistant Ready")
    time.sleep(2)
    clear_display()

    while True:
        audio_file = record_audio()
        text = await transcribe_audio(audio_file)
        os.remove(audio_file)

        if not text.strip():
            speak("Didn't catch that. Please try again.")
            continue

        if "exit" in text.lower() or "quit" in text.lower():
            speak("Goodbye!")
            clear_display()
            lcd.write_string("Goodbye!")
            time.sleep(2)
            break

        answer = ask_gemini(text)
        speak(answer)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nAssistant stopped.")
    finally:
        all_leds_off()
        clear_display()
        GPIO.cleanup()
