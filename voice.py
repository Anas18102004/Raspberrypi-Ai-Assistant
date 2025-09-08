import os
import asyncio
import json
import requests
from typing import Optional, Type, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
import sounddevice as sd
import numpy as np
from gtts import gTTS
import tempfile
import pygame
from enum import Enum
from pydantic import BaseModel, Field
from langchain.agents import Tool, AgentExecutor, create_structured_chat_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

# Load environment variables
load_dotenv()

# Initialize APIs
gemini_api_key = os.getenv("GEMINI_API_KEY")
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
openweather_api_key = os.getenv("OPENWEATHER_API_KEY")  # Add your OpenWeather API key

genai.configure(api_key=gemini_api_key)
deepgram = DeepgramClient(api_key=deepgram_api_key)

# Initialize pygame for audio playback
pygame.mixer.init()

class DeviceState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"

class VoiceAssistant:
    def __init__(self):
        self.state = DeviceState.IDLE
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            google_api_key=gemini_api_key
        )
        self.tools = self._setup_tools()
        self.agent = self._setup_agent()
        self.dg_connection = None
        self.sample_rate = 16000
        self.channels = 1
        self.is_running = False

    def _setup_tools(self):
        # Weather Tool
        class WeatherInput(BaseModel):
            location: str = Field(description="The city and state/country, e.g., 'San Francisco, US'")

        def get_weather(location: str) -> str:
            """Get the current weather for a location."""
            try:
                base_url = "http://api.openweathermap.org/data/2.5/weather"
                params = {
                    'q': location,
                    'appid': openweather_api_key,
                    'units': 'metric'
                }
                response = requests.get(base_url, params=params)
                data = response.json()
                
                if response.status_code == 200:
                    weather = data['weather'][0]['description']
                    temp = data['main']['temp']
                    return f"Current weather in {location}: {weather}, Temperature: {temp}°C"
                else:
                    return f"Could not get weather data for {location}: {data.get('message', 'Unknown error')}"
            except Exception as e:
                return f"Error getting weather: {str(e)}"

        # Google Search Tool
        class SearchInput(BaseModel):
            query: str = Field(description="search query to look up")

        def search_google(query: str) -> str:
            """Searches Google for the query. Useful for current events or information not in the knowledge base."""
            try:
                from googlesearch import search
                results = list(search(query, num=3, stop=3, pause=2))
                if results:
                    return f"Top search results for '{query}':\n" + "\n".join(results)
                return "No search results found."
            except Exception as e:
                return f"Error performing search: {str(e)}"

        # Create tool instances
        weather_tool = Tool(
            name="get_weather",
            func=get_weather,
            description="Useful for getting current weather information for a location.",
            args_schema=WeatherInput
        )

        search_tool = Tool(
            name="search_google",
            func=search_google,
            description="Useful for searching the internet for current information.",
            args_schema=SearchInput
        )

        return [weather_tool, search_tool]

    def _setup_agent(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful voice assistant named Krish. Keep your responses concise and natural for speech."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
            }
            | prompt
            | self.llm.bind(functions=[{"name": t.name, "description": t.description, "parameters": t.args} for t in self.tools])
            | OpenAIFunctionsAgentOutputParser()
        )

        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    async def text_to_speech(self, text: str):
        """Convert text to speech and play it"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                tts = gTTS(text=text, lang='en')
                tts.save(fp.name)
                
                pygame.mixer.music.load(fp.name)
                pygame.mixer.music.play()
                
                # Wait for the audio to finish playing
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
                
                os.unlink(fp.name)
        except Exception as e:
            print(f"Error in text_to_speech: {e}")

    async def process_voice_command(self, text: str):
        """Process the transcribed text and generate a response"""
        if not text.strip():
            return "I didn't catch that. Could you please repeat?"

        self.state = DeviceState.PROCESSING
        
        try:
            response = await self.agent.ainvoke({"input": text, "chat_history": []})
            return response["output"]
        except Exception as e:
            print(f"Error processing command: {e}")
            return "I'm sorry, I encountered an error processing your request."
        finally:
            self.state = DeviceState.IDLE

    async def start_voice_loop(self):
        """Main loop for voice interaction"""
        self.is_running = True
        
        # Configure Deepgram Live Transcription
        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=True,
            endpointing=300,
            utterance_end_ms="1000",
            vad_events=True,
        )

        def on_utterance_end(utterance_end, **kwargs):
            if utterance_end.get('utterance_end'):
                asyncio.create_task(self._process_utterance())

        def on_transcript(transcript, **kwargs):
            if 'channel' in transcript and 'alternatives' in transcript['channel']:
                utterance = transcript['channel']['alternatives'][0].get('transcript', '').strip()
                if utterance:
                    print(f"Heard: {utterance}")
                    self.current_utterance = utterance

        def on_error(error, **kwargs):
            print(f"Deepgram error: {error}")

        # Start Deepgram connection
        self.dg_connection = deepgram.listen.live.v("1").start(options)
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        self.dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        self.dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        # Start audio stream
        self.audio_queue = asyncio.Queue()
        self.current_utterance = ""
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(status, flush=True)
            self.audio_queue.put_nowait(indata.copy())

        stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=4096,
            dtype='int16',
            channels=self.channels,
            callback=audio_callback
        )

        with stream:
            print("Listening... (Press Ctrl+C to stop)")
            while self.is_running:
                try:
                    data = await self.audio_queue.get()
                    if self.dg_connection and self.dg_connection.is_alive():
                        self.dg_connection.send(data.tobytes())
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"Error in audio loop: {e}")

    async def _process_utterance(self):
        """Process the completed utterance"""
        if not self.current_utterance:
            return
            
        self.state = DeviceState.PROCESSING
        print(f"Processing: {self.current_utterance}")
        
        response = await self.process_voice_command(self.current_utterance)
        print(f"Response: {response}")
        
        await self.text_to_speech(response)
        self.current_utterance = ""
        self.state = DeviceState.IDLE

    async def stop(self):
        """Clean up resources"""
        self.is_running = False
        if self.dg_connection:
            await self.dg_connection.finish()
        pygame.mixer.quit()

async def main():
    assistant = VoiceAssistant()
    try:
        await assistant.start_voice_loop()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await assistant.stop()

if __name__ == "__main__":
    asyncio.run(main())
