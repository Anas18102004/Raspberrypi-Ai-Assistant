#!/usr/bin/env python3
"""
Krishna Voice Assistant for Raspberry Pi - Optimized Version
Advanced AI Assistant with Real-time Processing and Web Search Integration

Features:
- Real-time voice processing optimized for ARM architecture
- Intelligent context-aware responses
- Google Search integration for recent/unknown information
- Multi-modal interaction (voice + text + web)
- gTTS-based text-to-speech system
- Raspberry Pi hardware optimization
- Comprehensive conversation memory and context
"""

import os
import sys
import time
import threading
import queue
import logging
import json
import subprocess
import platform
import requests
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import traceback
import concurrent.futures
from dataclasses import dataclass
from urllib.parse import quote_plus
import tempfile
import io

# Core audio and ML libraries
import sounddevice as sd
import numpy as np
import wave
import whisper
from ctransformers import AutoModelForCausalLM

# gTTS for text-to-speech
from gtts import gTTS
import pygame

# Web search and parsing
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: BeautifulSoup4 not available, web search will be limited")

# Advanced VAD
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

@dataclass
class ConversationContext:
    """Track conversation context for intelligent responses"""
    recent_topics: List[str]
    user_preferences: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    last_search_query: Optional[str]
    last_search_time: Optional[datetime]
    current_session_start: datetime

class Config:
    """Configuration optimized for Raspberry Pi"""
    
    # Model path - adjust this for your Raspberry Pi setup
    TINYLLAMA_PATH = "/home/pi/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    # Audio Configuration - optimized for Pi
    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = np.float32
    
    # Real-time streaming parameters - reduced for Pi performance
    CHUNK_DURATION = 0.8  # Increased for better Pi performance
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
    OVERLAP_SIZE = int(CHUNK_SIZE * 0.25)
    
    # Advanced VAD parameters - tuned for Pi
    VAD_ENERGY_THRESHOLD = 0.008  # Slightly higher for Pi microphones
    VAD_SPECTRAL_THRESHOLD = 0.25
    VAD_MIN_SPEECH_DURATION = 0.5
    VAD_MAX_SILENCE_DURATION = 2.0
    VAD_HANGOVER_FRAMES = 3
    
    # Model Configuration - optimized for Pi
    WHISPER_MODEL = "tiny"  # Using tiny model for Pi performance
    MAX_TOKENS = 30  # Reduced for Pi
    TEMPERATURE = 0.3
    TOP_P = 0.8
    CONTEXT_LENGTH = 256  # Reduced for Pi memory
    REPETITION_PENALTY = 1.2
    
    # AI Assistant Configuration
    MAX_CONVERSATION_HISTORY = 15  # Reduced for Pi memory
    CONTEXT_WINDOW_MINUTES = 20
    
    # gTTS Configuration
    TTS_LANGUAGE = 'en'
    TTS_SLOW = False
    TTS_CACHE_DIR = '/tmp/krishna_tts_cache'
    
    # Web Search Configuration
    SEARCH_TRIGGERS = [
        "what's the latest", "recent news", "current", "today", "this week",
        "search for", "look up", "find information", "google", "search",
        "what happened", "breaking news", "update on", "recent developments"
    ]
    
    SEARCH_DOMAINS = [
        "news", "weather", "sports", "technology", "science", "politics",
        "stock market", "cryptocurrency", "current events"
    ]
    
    # Performance thresholds - adjusted for Pi
    MAX_PROCESSING_TIME = 5.0  # Increased for Pi
    PARALLEL_WORKERS = 2  # Reduced for Pi
    TARGET_RESPONSE_TIME = 2.0  # More realistic for Pi
    MAX_ACCEPTABLE_LATENCY = 4.0
    
    # Wake/sleep words
    ASSISTANT_WAKE_WORDS = ["hello krishna", "hey krishna", "krishna wake up", "wake up krishna"]
    ASSISTANT_SLEEP_WORDS = ["shutdown", "go to sleep", "sleep now", "power down"]
    
    # Emergency and priority words
    EMERGENCY_WORDS = ["emergency", "urgent", "help", "critical", "mayday"]
    PRIORITY_WORDS = ["important", "priority", "asap", "immediately"]

class GTTSHandler:
    """Optimized gTTS handler for Raspberry Pi"""
    
    def __init__(self):
        self.cache_dir = Config.TTS_CACHE_DIR
        self.setup_cache()
        self.setup_pygame()
        self.cache = {}  # In-memory cache for recent TTS
        
    def setup_cache(self):
        """Setup TTS cache directory"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def setup_pygame(self):
        """Initialize pygame mixer for audio playback"""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.pygame_available = True
            print("✓ pygame audio mixer initialized")
        except Exception as e:
            print(f"pygame mixer initialization failed: {e}")
            self.pygame_available = False
    
    def get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def speak(self, text: str):
        """Speak text using gTTS with caching"""
        if not text.strip():
            return
            
        try:
            cache_key = self.get_cache_key(text)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.mp3")
            
            # Check if cached file exists
            if not os.path.exists(cache_file):
                # Generate new TTS
                tts = gTTS(
                    text=text, 
                    lang=Config.TTS_LANGUAGE, 
                    slow=Config.TTS_SLOW
                )
                tts.save(cache_file)
            
            # Play audio
            if self.pygame_available:
                self.play_with_pygame(cache_file)
            else:
                self.play_with_system(cache_file)
                
        except Exception as e:
            print(f"TTS error: {e}")
            print(f"[Krishna]: {text}")
    
    def play_with_pygame(self, audio_file: str):
        """Play audio using pygame"""
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
        except Exception as e:
            print(f"pygame playback error: {e}")
            self.play_with_system(audio_file)
    
    def play_with_system(self, audio_file: str):
        """Fallback to system audio player"""
        try:
            # Try different audio players available on Pi
            players = ['mpg123', 'omxplayer', 'aplay', 'paplay']
            
            for player in players:
                if subprocess.run(['which', player], 
                                capture_output=True).returncode == 0:
                    if player == 'omxplayer':
                        subprocess.run([player, '-o', 'local', audio_file], 
                                     capture_output=True)
                    else:
                        subprocess.run([player, audio_file], 
                                     capture_output=True)
                    return
                    
            print("No audio player found, TTS output to console only")
            
        except Exception as e:
            print(f"System audio playback error: {e}")
    
    def cleanup_old_cache(self, max_age_hours: int = 24):
        """Clean up old cache files"""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.getctime(filepath) < cutoff_time:
                    os.remove(filepath)
        except Exception as e:
            print(f"Cache cleanup error: {e}")

class WebSearchEngine:
    """Intelligent web search engine for knowledge augmentation"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux armv7l) AppleWebKit/537.36'
        })
        self.last_search_time = {}  # Rate limiting
        
    def should_search(self, query: str, context: ConversationContext) -> bool:
        """Determine if query requires web search"""
        query_lower = query.lower()
        
        # Check for explicit search triggers
        for trigger in Config.SEARCH_TRIGGERS:
            if trigger in query_lower:
                return True
        
        # Check for time-sensitive queries
        time_indicators = ["today", "now", "current", "latest", "recent", "this week", "this month"]
        if any(indicator in query_lower for indicator in time_indicators):
            return True
        
        # Check for domain-specific queries that need current info
        for domain in Config.SEARCH_DOMAINS:
            if domain in query_lower:
                return True
        
        # Check for factual questions that might need verification
        question_starters = ["what is", "who is", "when did", "where is", "how many"]
        if any(query_lower.startswith(starter) for starter in question_starters):
            return True
        
        return False
    
    def search_google(self, query: str, num_results: int = 2) -> List[Dict[str, str]]:
        """Search Google and return structured results"""
        try:
            # Rate limiting
            now = time.time()
            if query in self.last_search_time:
                if now - self.last_search_time[query] < 15:  # 15 second cooldown for Pi
                    return []
            
            self.last_search_time[query] = now
            
            # Use DuckDuckGo as more reliable alternative
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            
            response = self.session.get(search_url, timeout=8)  # Increased timeout for Pi
            response.raise_for_status()
            
            if not BS4_AVAILABLE:
                # Simple fallback without parsing
                return [{
                    "title": "Search Results Available",
                    "snippet": f"Found information about: {query}",
                    "url": search_url
                }]
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Parse DuckDuckGo results
            for result in soup.find_all('div', class_='result')[:num_results]:
                try:
                    title_elem = result.find('a', class_='result__a')
                    snippet_elem = result.find('a', class_='result__snippet')
                    
                    if title_elem and snippet_elem:
                        results.append({
                            "title": title_elem.get_text().strip(),
                            "snippet": snippet_elem.get_text().strip(),
                            "url": title_elem.get('href', '')
                        })
                except Exception:
                    continue
            
            return results
            
        except Exception as e:
            logging.error(f"Search error: {e}")
            return []
    
    def get_current_time_info(self) -> str:
        """Get comprehensive current time information"""
        now = datetime.now()
        return f"Current time: {now.strftime('%I:%M %p')}, Date: {now.strftime('%A, %B %d, %Y')}"

class IntelligentResponseGenerator:
    """Advanced response generation with context awareness"""
    
    def __init__(self, llm, search_engine: WebSearchEngine):
        self.llm = llm
        self.search_engine = search_engine
        self.conversation_context = ConversationContext(
            recent_topics=[],
            user_preferences={},
            conversation_history=[],
            last_search_query=None,
            last_search_time=None,
            current_session_start=datetime.now()
        )
    
    def update_context(self, user_input: str, assistant_response: str):
        """Update conversation context"""
        self.conversation_context.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response
        })
        
        # Keep only recent history
        if len(self.conversation_context.conversation_history) > Config.MAX_CONVERSATION_HISTORY:
            self.conversation_context.conversation_history.pop(0)
        
        # Extract topics (simple keyword extraction)
        words = user_input.lower().split()
        topics = [word for word in words if len(word) > 4 and word.isalpha()]
        self.conversation_context.recent_topics.extend(topics[-3:])  # Last 3 meaningful words
        
        # Keep recent topics list manageable
        if len(self.conversation_context.recent_topics) > 15:  # Reduced for Pi
            self.conversation_context.recent_topics = self.conversation_context.recent_topics[-15:]
    
    def generate_enhanced_response(self, user_input: str) -> str:
        """Generate contextually aware response with web search integration"""
        try:
            # Check if we need to search
            if self.search_engine.should_search(user_input, self.conversation_context):
                return self.generate_search_enhanced_response(user_input)
            else:
                return self.generate_local_response(user_input)
        
        except Exception as e:
            logging.error(f"Response generation error: {e}")
            return "I'm experiencing some technical difficulties. Could you please try again?"
    
    def generate_search_enhanced_response(self, user_input: str) -> str:
        """Generate response with web search integration"""
        try:
            # Perform search
            search_results = self.search_engine.search_google(user_input)
            
            if search_results:
                # Combine search results with local knowledge
                search_context = ""
                for i, result in enumerate(search_results[:1]):  # Use only 1 result for Pi
                    search_context += f"Recent info: {result['title']} - {result['snippet'][:150]}... "
                
                # Create enhanced prompt - simplified for Pi
                prompt = f"""Krishna assistant with current info.
User: {user_input}
Context: {search_context}
Respond helpfully in under 20 words:
Krishna:"""
                
                response = self.llm(prompt)
                self.conversation_context.last_search_query = user_input
                self.conversation_context.last_search_time = datetime.now()
                
            else:
                # Fallback to local response if search fails
                response = self.generate_local_response(user_input)
                response += " (Could not access online information.)"
            
            return self.clean_response(response)
            
        except Exception as e:
            logging.error(f"Search-enhanced response error: {e}")
            return self.generate_local_response(user_input)
    
    def generate_local_response(self, user_input: str) -> str:
        """Generate response using local LLM with conversation context"""
        try:
            context_summary = self.get_context_summary()
            
            # Create contextual prompt - simplified for Pi
            prompt = f"""Krishna AI assistant.
Context: {context_summary}
User: {user_input}
Respond naturally in under 20 words:
Krishna:"""
            
            response = self.llm(prompt)
            return self.clean_response(response)
            
        except Exception as e:
            logging.error(f"Local response error: {e}")
            return "I need a moment to process that. Could you repeat your question?"
    
    def get_context_summary(self) -> str:
        """Get summary of recent conversation context"""
        if not self.conversation_context.conversation_history:
            return "New conversation"
        
        recent_exchanges = self.conversation_context.conversation_history[-2:]  # Reduced for Pi
        summary_parts = []
        
        for exchange in recent_exchanges:
            user_text = exchange['user'][:30] + "..." if len(exchange['user']) > 30 else exchange['user']
            summary_parts.append(f"User: {user_text}")
        
        return " | ".join(summary_parts)
    
    def clean_response(self, response: str) -> str:
        """Clean and validate the response"""
        if isinstance(response, str):
            response = response.strip()
            
            # Remove common prefixes
            prefixes = ["Krishna:", "Assistant:", "AI:", "Response:", "Answer:"]
            for prefix in prefixes:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            
            # Limit word count for Pi performance
            words = response.split()[:20]  # Max 20 words for Pi
            response = ' '.join(words)
            
            # Ensure proper ending
            if response and response[-1] not in '.!?':
                response += '.'
            
            # Validate minimum quality
            if len(response.split()) < 2:
                return "I need more context to help you."
            
            return response
        
        return "I'm having trouble with that request."

class AdvancedVAD:
    """Voice Activity Detection optimized for Raspberry Pi"""
    
    def __init__(self, sample_rate=Config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.frame_length = int(0.025 * sample_rate)
        self.hop_length = int(0.01 * sample_rate)
        self.n_mfcc = 13
        self.n_fft = 512
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speech_active = False
        
    def analyze_chunk(self, audio_chunk: np.ndarray) -> Dict:
        """Analyze audio chunk for speech activity - Pi optimized"""
        if len(audio_chunk) == 0:
            return {"is_speech": False, "confidence": 0.0, "energy": 0.0}
        
        energy = np.mean(audio_chunk ** 2)
        
        # Simplified VAD for Pi performance
        if LIBROSA_AVAILABLE:
            try:
                zcr = np.mean(librosa.feature.zero_crossing_rate(audio_chunk, 
                                                               frame_length=self.frame_length,
                                                               hop_length=self.hop_length))
                confidence = energy * 2.0 + (1.0 if zcr > 0.01 and zcr < 0.3 else 0.0)
            except Exception:
                zcr = np.mean(np.diff(np.signbit(audio_chunk)))
                confidence = energy * 3.0
        else:
            zcr = np.mean(np.diff(np.signbit(audio_chunk)))
            confidence = energy * 3.0 if energy > Config.VAD_ENERGY_THRESHOLD else 0.0
        
        is_speech = confidence > Config.VAD_SPECTRAL_THRESHOLD
        
        return {
            "is_speech": is_speech,
            "confidence": min(confidence, 1.0),
            "energy": energy,
            "zcr": zcr
        }
    
    def update_state(self, vad_result: Dict) -> bool:
        """Update speech state with hangover logic"""
        if vad_result["is_speech"]:
            self.speech_frames += 1
            self.silence_frames = 0
            if not self.is_speech_active and self.speech_frames >= 2:  # Reduced for Pi
                self.is_speech_active = True
                return True
        else:
            self.silence_frames += 1
            if self.is_speech_active:
                if self.silence_frames >= Config.VAD_HANGOVER_FRAMES:
                    self.is_speech_active = False
                    self.speech_frames = 0
                    return True
        return False

class RealTimeAudioProcessor:
    """Real-time streaming audio processor optimized for Pi"""
    
    def __init__(self, callback, sample_rate=Config.SAMPLE_RATE):
        self.callback = callback
        self.sample_rate = sample_rate
        self.chunk_size = Config.CHUNK_SIZE
        self.vad = AdvancedVAD(sample_rate)
        self.recording_buffer = []
        self.is_recording = False
        self.last_speech_time = 0
        self.chunk_processing_times = []
        
    def audio_callback(self, indata, frames, time_info, status):
        """Real-time audio callback - Pi optimized"""
        try:
            if status:
                logging.warning(f"Audio callback status: {status}")
            
            audio_chunk = indata.flatten().copy()
            energy = np.mean(audio_chunk ** 2)
            
            # Skip very quiet audio to save Pi resources
            if energy < Config.VAD_ENERGY_THRESHOLD * 0.05:
                return
            
            try:
                self.process_chunk_async(audio_chunk)
            except Exception as e:
                logging.error(f"Chunk processing error: {e}")
        except Exception as e:
            logging.error(f"Audio callback error: {e}")
    
    def process_chunk_async(self, audio_chunk: np.ndarray):
        """Process audio chunk asynchronously - Pi optimized"""
        start_time = time.time()
        vad_result = self.vad.analyze_chunk(audio_chunk)
        state_changed = self.vad.update_state(vad_result)
        current_time = time.time()
        
        if vad_result["is_speech"]:
            self.last_speech_time = current_time
            if not self.is_recording:
                self.is_recording = True
                self.recording_buffer = [audio_chunk]
                print(f"\rListening... (conf: {vad_result['confidence']:.2f})", end="", flush=True)
            else:
                self.recording_buffer.append(audio_chunk)
                
                # Limit buffer size for Pi memory
                if len(self.recording_buffer) > 100:  # About 10 seconds at current settings
                    self.finalize_recording()
                    
        elif self.is_recording:
            time_since_speech = current_time - self.last_speech_time
            if time_since_speech < Config.VAD_MAX_SILENCE_DURATION:
                self.recording_buffer.append(audio_chunk)
            else:
                self.finalize_recording()
        
        processing_time = time.time() - start_time
        self.chunk_processing_times.append(processing_time)
        if len(self.chunk_processing_times) > 50:  # Reduced for Pi memory
            self.chunk_processing_times.pop(0)
    
    def finalize_recording(self):
        """Finalize recording and send for processing"""
        if not self.recording_buffer:
            return
        
        full_audio = np.concatenate(self.recording_buffer)
        duration = len(full_audio) / self.sample_rate
        
        if duration >= Config.VAD_MIN_SPEECH_DURATION:
            print(f"\rProcessing speech ({duration:.1f}s)...")
            threading.Thread(target=self.callback, args=(full_audio,), daemon=True).start()
        
        self.is_recording = False
        self.recording_buffer = []
    
    def start_stream(self):
        """Start real-time audio stream"""
        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                samplerate=self.sample_rate,
                channels=Config.CHANNELS,
                dtype=Config.DTYPE,
                blocksize=self.chunk_size,
                latency='low'
            )
            self.stream.start()
            print("Krishna AI Assistant listening - Optimized for Raspberry Pi...")
            return True
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            return False
    
    def stop_stream(self):
        """Stop audio stream"""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
    
    def get_performance_stats(self) -> Dict:
        """Get processing performance statistics"""
        if not self.chunk_processing_times:
            return {"avg_processing_time": 0, "max_processing_time": 0}
        return {
            "avg_processing_time": np.mean(self.chunk_processing_times),
            "max_processing_time": np.max(self.chunk_processing_times),
            "chunk_count": len(self.chunk_processing_times)
        }

class KrishnaRaspberryPiAssistant:
    """Complete Krishna AI assistant optimized for Raspberry Pi"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Check systems
        self.check_audio_devices()
        
        # Core systems
        self.current_mode = "TEST"
        self.is_sleeping = False
        self.is_running = True
        self.audio_processor = None
        
        # Advanced AI components
        self.search_engine = WebSearchEngine()
        self.response_generator = None  # Initialize after models load
        self.tts_handler = GTTSHandler()
        
        # Processing pipeline - reduced for Pi
        self.transcription_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # Performance tracking
        self.response_times = []
        self.successful_interactions = 0
        self.failed_interactions = 0
        self.search_queries_made = 0
        
        # Initialize AI systems
        self.initialize_models()
        
        # Initialize intelligent response generator
        self.response_generator = IntelligentResponseGenerator(self.llm, self.search_engine)
        
        print("Krishna Assistant for Raspberry Pi - Ready!")
        print("Features: Voice processing, Intelligent conversations, Web search, gTTS")
    
    def setup_logging(self):
        """Setup logging for Pi"""
        log_dir = "/tmp/krishna_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}/krishna_pi.log'),
                logging.StreamHandler()
            ]
        )
    
    def check_audio_devices(self):
        """Check and configure audio devices for Pi"""
        try:
            devices = sd.query_devices()
            print("Available audio input devices:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"  {i}: {device['name']}")
            
            # Try to set a reasonable default for Pi
            default_input = sd.default.device[0]
            if default_input is None:
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0 and 'usb' in device['name'].lower():
                        sd.default.device[0] = i
                        print(f"Set default input device to: {device['name']}")
                        break
        except Exception as e:
            print(f"Audio device check failed: {e}")
    
    def initialize_models(self):
        """Initialize AI models for Pi"""
        print("Loading AI models for Raspberry Pi (this will take a moment)...")
        
        # Load Whisper with tiny model
        try:
            self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
            print(f"✓ Whisper {Config.WHISPER_MODEL} loaded (Pi optimized)")
        except Exception as e:
            print(f"✗ Whisper loading failed: {e}")
            raise
        
        # Load TinyLlama with Pi optimizations
        try:
            if not os.path.exists(Config.TINYLLAMA_PATH):
                print(f"Model not found at: {Config.TINYLLAMA_PATH}")
                print("Please download the model to the specified path")
                print("Or update TINYLLAMA_PATH in Config class")
                raise FileNotFoundError(f"Model not found: {Config.TINYLLAMA_PATH}")
            
            self.llm = AutoModelForCausalLM.from_pretrained(
                Config.TINYLLAMA_PATH,
                max_new_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                context_length=Config.CONTEXT_LENGTH,
                threads=2,  # Reduced for Pi
                repetition_penalty=Config.REPETITION_PENALTY,
                stop=["Human:", "User:", "Krishna:", "\n"]
            )
            print("✓ TinyLlama loaded (Pi optimized)")
        except Exception as e:
            print(f"✗ TinyLlama loading failed: {e}")
            raise
    
    def speak_intelligent(self, text: str):
        """Speak text using gTTS"""
        try:
            self.tts_handler.speak(text)
        except Exception as e:
            print(f"TTS error: {e}")
            print(f"[Krishna]: {text}")
    
    def process_intelligent_audio(self, audio_data: np.ndarray):
        """Process audio with Pi-optimized AI capabilities"""
        start_time = time.time()
        
        try:
            # Transcription
            transcription_future = self.transcription_executor.submit(
                self.transcribe_audio_optimized, audio_data
            )
            
            try:
                text = transcription_future.result(timeout=Config.MAX_PROCESSING_TIME)
            except concurrent.futures.TimeoutError:
                print("Transcription timeout - please try again")
                self.failed_interactions += 1
                return
            
            if not text:
                return
            
            print(f"You said: \"{text}\"")
            
            # Handle system commands first
            if self.handle_system_commands(text):
                return
            
            # Generate intelligent response
            if not self.is_sleeping:
                try:
                    # Use the intelligent response generator
                    response = self.response_generator.generate_enhanced_response(text)
                    
                    if response:
                        print(f"Krishna: {response}")
                        self.speak_intelligent(response)
                        
                        # Update conversation context
                        self.response_generator.update_context(text, response)
                        
                        # Track performance
                        elapsed = time.time() - start_time
                        self.response_times.append(elapsed)
                        self.successful_interactions += 1
                        
                        # Check if search was used
                        if self.response_generator.conversation_context.last_search_time:
                            if self.response_generator.conversation_context.last_search_time >= datetime.now() - timedelta(seconds=10):
                                self.search_queries_made += 1
                        
                        print(f"Response time: {elapsed:.3f}s")
                        
                except Exception as e:
                    print(f"Response generation error: {e}")
                    self.speak_intelligent("I'm having some processing difficulties. Please try again.")
                    self.failed_interactions += 1
        
        except Exception as e:
            print(f"Audio processing error: {e}")
            self.failed_interactions += 1
    
    def transcribe_audio_optimized(self, audio_data: np.ndarray) -> Optional[str]:
        """Optimized transcription for Pi"""
        try:
            # Use temporary file in RAM disk if available
            temp_dir = "/tmp" if os.path.exists("/tmp") else "."
            temp_file = os.path.join(temp_dir, f"temp_audio_{threading.get_ident()}.wav")
            
            # Convert and save audio
            audio_int16 = (audio_data * 32767).astype(np.int16)
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(Config.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(Config.SAMPLE_RATE)
                wf.writeframes(audio_int16.tobytes())
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(
                temp_file, 
                fp16=False,
                verbose=False,
                language='en'
            )
            
            text = result["text"].strip()
            
            # Cleanup temp file
            try:
                os.remove(temp_file)
            except:
                pass
            
            # Validate transcription quality
            if len(text) < 2 or len(text.split()) < 1:
                return None
            
            return text
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
    
    def handle_system_commands(self, text: str) -> bool:
        """Handle system-level commands optimized for Pi"""
        text_lower = text.lower().strip()
        
        # Emergency handling first
        if any(word in text_lower for word in Config.EMERGENCY_WORDS):
            self.speak_intelligent("Emergency protocols activated. How can I assist you?")
            return True
        
        # Wake commands
        for wake_word in Config.ASSISTANT_WAKE_WORDS:
            if wake_word in text_lower:
                if self.is_sleeping:
                    self.is_sleeping = False
                    print("System awakening...")
                    self.speak_intelligent("Hello! I'm awake and ready to assist.")
                    return True
                else:
                    self.speak_intelligent("I'm already active and listening.")
                    return True
        
        # Sleep commands
        for sleep_word in Config.ASSISTANT_SLEEP_WORDS:
            if sleep_word in text_lower:
                self.is_sleeping = True
                print("Entering sleep mode...")
                self.speak_intelligent("Entering sleep mode. Say hello Krishna to wake me up.")
                return True
        
        # Mode switching
        if "assistant mode" in text_lower:
            if self.switch_mode("ASSISTANT"):
                self.speak_intelligent("Assistant mode activated. Real-time processing enabled.")
                return True
        elif "test mode" in text_lower:
            if self.switch_mode("TEST"):
                self.speak_intelligent("Test mode activated.")
                return True
        
        # System status and information
        if "system status" in text_lower or "health check" in text_lower:
            self.provide_system_status()
            return True
        
        if "capabilities" in text_lower or "what can you do" in text_lower:
            self.explain_capabilities()
            return True
        
        # Pi-specific commands
        if "temperature" in text_lower and "pi" in text_lower:
            self.get_pi_temperature()
            return True
            
        if "memory usage" in text_lower:
            self.get_memory_usage()
            return True
        
        return False
    
    def get_pi_temperature(self):
        """Get Raspberry Pi CPU temperature"""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_raw = f.read().strip()
                temp_c = int(temp_raw) / 1000
                temp_f = (temp_c * 9/5) + 32
                
            status_msg = f"Pi CPU temperature: {temp_c:.1f}°C ({temp_f:.1f}°F)"
            print(status_msg)
            
            if temp_c > 70:
                self.speak_intelligent(f"CPU temperature is {temp_c:.0f} degrees Celsius. Consider cooling.")
            else:
                self.speak_intelligent(f"CPU temperature is {temp_c:.0f} degrees Celsius. Normal.")
                
        except Exception as e:
            print(f"Temperature check error: {e}")
            self.speak_intelligent("Unable to check CPU temperature.")
    
    def get_memory_usage(self):
        """Get memory usage information"""
        try:
            # Get memory info
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) // 1024
            mem_available = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1]) // 1024
            mem_used = mem_total - mem_available
            mem_percent = (mem_used / mem_total) * 100
            
            status_msg = f"Memory usage: {mem_used}MB / {mem_total}MB ({mem_percent:.1f}%)"
            print(status_msg)
            self.speak_intelligent(f"Memory usage is {mem_percent:.0f} percent.")
            
        except Exception as e:
            print(f"Memory check error: {e}")
            self.speak_intelligent("Unable to check memory usage.")
    
    def provide_system_status(self):
        """Provide Pi-optimized system status"""
        avg_response = np.mean(self.response_times) if self.response_times else 0
        success_rate = self.successful_interactions / max(1, self.successful_interactions + self.failed_interactions)
        uptime = datetime.now() - self.response_generator.conversation_context.current_session_start
        
        status_msg = f"""Pi System Status: All systems operational. 
        Success rate: {success_rate:.1%}. 
        Average response time: {avg_response:.2f}s. 
        Search queries: {self.search_queries_made}. 
        Session uptime: {str(uptime).split('.')[0]}."""
        
        print(status_msg)
        self.speak_intelligent(f"All systems operational. Success rate {success_rate:.0%}. Response time {avg_response:.1f} seconds.")
    
    def explain_capabilities(self):
        """Explain Pi system capabilities"""
        capabilities_msg = """Krishna AI Assistant for Raspberry Pi:
        - Real-time voice processing optimized for ARM
        - Intelligent conversations with context memory
        - Web search integration for current information
        - gTTS text-to-speech system
        - Pi hardware monitoring
        - Energy-efficient operation"""
        
        print(capabilities_msg)
        self.speak_intelligent("I offer voice processing, intelligent conversations, web search, and Pi system monitoring.")
    
    def switch_mode(self, new_mode: str) -> bool:
        """Switch between operating modes"""
        if new_mode.upper() not in ["ASSISTANT", "TEST"]:
            return False
        
        old_mode = self.current_mode
        self.current_mode = new_mode.upper()
        
        if old_mode != self.current_mode:
            print(f"Mode switched: {old_mode} -> {self.current_mode}")
            
            if self.current_mode == "ASSISTANT":
                self.start_pi_assistant_mode()
            else:
                self.stop_assistant_mode()
        
        return True
    
    def start_pi_assistant_mode(self):
        """Start Pi-optimized assistant mode"""
        if self.audio_processor:
            return
        
        print("Initializing Pi AI Assistant Mode...")
        print("Features: Real-time voice, Context awareness, Web search, gTTS")
        
        self.audio_processor = RealTimeAudioProcessor(
            callback=self.process_intelligent_audio
        )
        
        success = self.audio_processor.start_stream()
        if success:
            self.speak_intelligent("Assistant mode active on Raspberry Pi.")
        else:
            print("Failed to start assistant mode")
    
    def stop_assistant_mode(self):
        """Stop assistant mode"""
        if self.audio_processor:
            self.audio_processor.stop_stream()
            self.audio_processor = None
            print("Assistant mode stopped")
    
    def run_pi_session(self):
        """Main interactive session optimized for Pi"""
        print("\n" + "="*60)
        print("KRISHNA AI ASSISTANT - RASPBERRY PI EDITION")
        print("="*60)
        print("Features:")
        print("   - Voice processing optimized for ARM")
        print("   - Intelligent conversations with gTTS")
        print("   - Web search integration")
        print("   - Pi hardware monitoring")
        print("="*60)
        print("Commands: [a] Assistant | [t] Test | [s] Status | [h] Help | [q] Quit")
        print("="*60)
        
        try:
            while self.is_running:
                if self.current_mode == "ASSISTANT":
                    if self.is_sleeping:
                        print("\n🟡 ASSISTANT MODE - SLEEPING")
                        print("   System monitoring for wake commands...")
                    else:
                        print(f"\n🟢 PI ASSISTANT MODE - ACTIVE")
                        print("   Real-time voice processing with gTTS")
                        if self.audio_processor:
                            perf = self.audio_processor.get_performance_stats()
                            print(f"   Audio processing: {perf['avg_processing_time']*1000:.1f}ms avg")
                    
                    user_input = input(f"\n[Assistant Mode]: ").strip().lower()
                    
                    if user_input == 'q':
                        break
                    elif user_input == 't':
                        self.switch_mode("TEST")
                    elif user_input == 's':
                        self.print_pi_stats()
                    elif user_input == 'h':
                        self.print_pi_help()
                    elif user_input == 'clear':
                        self.clear_conversation_history()
                    elif user_input == 'temp':
                        self.get_pi_temperature()
                    elif user_input == 'mem':
                        self.get_memory_usage()
                
                else:
                    print(f"\n🔵 TEST MODE - MANUAL OPERATION")
                    user_input = input("\n[ENTER] Record | [a] Assistant | [s] Status | [h] Help | [q] Quit: ").strip().lower()
                    
                    if user_input == 'q':
                        break
                    elif user_input == 'a':
                        self.switch_mode("ASSISTANT")
                    elif user_input == 's':
                        self.print_pi_stats()
                    elif user_input == 'h':
                        self.print_pi_help()
                    elif user_input == '':
                        self.manual_record_and_process()
                    elif user_input.startswith('text:'):
                        # Text input mode for testing
                        test_text = user_input[5:].strip()
                        if test_text:
                            print(f"Processing text: \"{test_text}\"")
                            response = self.response_generator.generate_enhanced_response(test_text)
                            print(f"Krishna: {response}")
                            self.speak_intelligent(response)
                    elif user_input == 'temp':
                        self.get_pi_temperature()
                    elif user_input == 'mem':
                        self.get_memory_usage()
        
        except KeyboardInterrupt:
            print("\nShutdown initiated...")
        finally:
            self.pi_shutdown()
    
    def manual_record_and_process(self):
        """Manual recording for test mode"""
        print("🎙 Recording for 4 seconds (Pi optimized)...")
        
        recording = sd.rec(
            int(4 * Config.SAMPLE_RATE),
            samplerate=Config.SAMPLE_RATE,
            channels=Config.CHANNELS,
            dtype=Config.DTYPE
        )
        sd.wait()
        
        audio_data = recording.flatten()
        print("Processing with Pi AI capabilities...")
        self.process_intelligent_audio(audio_data)
    
    def clear_conversation_history(self):
        """Clear conversation history and context"""
        self.response_generator.conversation_context.conversation_history = []
        self.response_generator.conversation_context.recent_topics = []
        print("Conversation history cleared.")
        self.speak_intelligent("Conversation history cleared.")
    
    def print_pi_help(self):
        """Print Pi-specific help information"""
        help_text = """
🤖 KRISHNA AI ASSISTANT - RASPBERRY PI HELP

VOICE COMMANDS:
• "Hello Krishna" / "Hey Krishna" - Wake up system
• "Shutdown" / "Go to sleep" - Put system to sleep
• "Assistant mode" / "Test mode" - Switch modes
• "System status" - Get system status
• "Temperature Pi" - Check CPU temperature
• "Memory usage" - Check RAM usage

CONSOLE COMMANDS:
• [a] - Switch to Assistant Mode
• [t] - Switch to Test Mode  
• [s] - Show system statistics
• [h] - Show this help
• [q] - Quit application
• [temp] - Check Pi temperature
• [mem] - Check memory usage
• [clear] - Clear conversation history
• text: <message> - Process text input

PI OPTIMIZATIONS:
• Reduced model size (Whisper tiny)
• gTTS with local caching
• Memory usage monitoring
• CPU temperature monitoring
• ARM-optimized processing
        """
        print(help_text)
    
    def print_pi_stats(self):
        """Print Pi-specific system statistics"""
        print(f"\n{'='*50}")
        print(f"📊 RASPBERRY PI SYSTEM STATISTICS")
        print(f"{'='*50}")
        
        # Basic stats
        success_rate = self.successful_interactions / max(1, self.successful_interactions + self.failed_interactions)
        avg_response = np.mean(self.response_times) if self.response_times else 0
        uptime = datetime.now() - self.response_generator.conversation_context.current_session_start
        
        print(f"System Mode: {self.current_mode}")
        print(f"System State: {'SLEEPING' if self.is_sleeping else 'ACTIVE'}")
        print(f"Session Uptime: {str(uptime).split('.')[0]}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Successful Interactions: {self.successful_interactions}")
        print(f"Failed Interactions: {self.failed_interactions}")
        
        # Performance metrics
        if self.response_times:
            max_response = np.max(self.response_times)
            min_response = np.min(self.response_times)
            
            print(f"\nPerformance Metrics:")
            print(f"Average Response Time: {avg_response:.3f}s")
            print(f"Best Response Time: {min_response:.3f}s")
            print(f"Worst Response Time: {max_response:.3f}s")
        
        # AI-specific stats
        print(f"\nAI Features:")
        print(f"Web Search Queries: {self.search_queries_made}")
        print(f"Conversation Turns: {len(self.response_generator.conversation_context.conversation_history)}")
        
        # Pi hardware stats
        try:
            # CPU temperature
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp_c = int(f.read().strip()) / 1000
            print(f"\nPi Hardware:")
            print(f"CPU Temperature: {temp_c:.1f}°C")
            
            # Memory usage
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) // 1024
            mem_available = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1]) // 1024
            mem_used = mem_total - mem_available
            mem_percent = (mem_used / mem_total) * 100
            print(f"Memory Usage: {mem_used}MB / {mem_total}MB ({mem_percent:.1f}%)")
            
        except Exception as e:
            print(f"Hardware stats error: {e}")
        
        print(f"{'='*50}")
    
    def pi_shutdown(self):
        """Pi-optimized shutdown sequence"""
        print("\nInitiating Pi system shutdown...")
        self.is_running = False
        
        # Stop audio processing
        self.stop_assistant_mode()
        
        # Cleanup TTS cache if needed
        try:
            self.tts_handler.cleanup_old_cache()
        except:
            pass
        
        # Shutdown thread pools
        print("Shutting down processing threads...")
        self.transcription_executor.shutdown(wait=True)
        self.llm_executor.shutdown(wait=True)
        
        # Final statistics
        if self.response_times:
            avg_response = np.mean(self.response_times)
            success_rate = self.successful_interactions / max(1, self.successful_interactions + self.failed_interactions)
            uptime = datetime.now() - self.response_generator.conversation_context.current_session_start
            
            print(f"\nFinal Pi Session Statistics:")
            print(f"Total uptime: {str(uptime).split('.')[0]}")
            print(f"Interactions processed: {self.successful_interactions}")
            print(f"Success rate: {success_rate:.1%}")
            print(f"Average response time: {avg_response:.2f}s")
            print(f"Web searches performed: {self.search_queries_made}")
        
        print("Krishna AI Assistant for Raspberry Pi shutdown complete.")
        print("Thank you for using Pi-optimized AI technology!")

def check_pi_requirements():
    """Check Pi system requirements and dependencies"""
    print("Checking Raspberry Pi system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Warning: Python 3.7+ recommended for optimal performance")
    
    # Check for required packages
    required_packages = [
        ('sounddevice', 'sounddevice'),
        ('numpy', 'numpy'),
        ('whisper', 'openai-whisper'),
        ('ctransformers', 'ctransformers'),
        ('gtts', 'gTTS'),
        ('pygame', 'pygame'),
        ('requests', 'requests')
    ]
    
    missing_packages = []
    for package, pip_name in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Install with: pip install {pip_name}")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\nTo install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Check audio system
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices:
            print("Warning: No audio input devices found")
        else:
            print(f"✓ Found {len(input_devices)} audio input device(s)")
    except Exception as e:
        print(f"Audio system check failed: {e}")
    
    # Check available memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) // 1024
        print(f"✓ Available RAM: {mem_total}MB")
        
        if mem_total < 1024:
            print("Warning: Less than 1GB RAM available. Performance may be limited.")
    except:
        print("Could not check memory information")
    
    return len(missing_packages) == 0

def main():
    """Main entry point for Pi Krishna assistant"""
    print("Initializing Krishna AI Assistant for Raspberry Pi...")
    
    # Check system requirements
    if not check_pi_requirements():
        print("\nPlease install missing requirements before running.")
        return
    
    try:
        assistant = KrishnaRaspberryPiAssistant()
        assistant.run_pi_session()
    
    except KeyboardInterrupt:
        print("\nEmergency shutdown initiated")
    except Exception as e:
        print(f"Critical system error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
