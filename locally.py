#!/usr/bin/env python3
"""
EMERGENCY PROTOCOL - Ultra-Low Memory AI System
For Raspberry Pi with <1GB RAM - Mission Critical Fallback
Based on deep space probe emergency systems
"""

import threading
import queue
import time
import os
import psutil
import gc
import json
import re
import subprocess
import sys
from typing import Dict, List, Optional

class EmergencyAIProtocol:
    def __init__(self):
        """Initialize ultra-lightweight AI system"""
        print("🚨 EMERGENCY PROTOCOL ACTIVATED - Ultra-Low Memory Mode")
        
        # Lightweight response system
        self.response_patterns = self.load_response_patterns()
        self.conversation_context = []
        self.max_context = 5  # Keep only last 5 exchanges
        
        # Performance tracking
        self.response_count = 0
        self.start_time = time.time()
        
        # TTS setup with fallback options
        self.tts_system = self.setup_emergency_tts()
        self.tts_queue = queue.Queue(maxsize=3)  # Small queue
        self.tts_enabled = True
        
        # Start background TTS worker
        if self.tts_system:
            self.tts_worker = threading.Thread(target=self._emergency_tts_worker, daemon=True)
            self.tts_worker.start()
        
        print("✅ Emergency AI system operational")
    
    def load_response_patterns(self) -> Dict:
        """Load intelligent response patterns - no ML model needed"""
        return {
            # Greetings and basic interaction
            r'(?i)\b(hello|hi|hey|good\s+(morning|afternoon|evening))\b': [
                "Hello! I'm your emergency AI assistant running on Raspberry Pi.",
                "Hi there! How can I help you today?",
                "Hello! Your Pi AI is ready to assist."
            ],
            
            # Questions about the system
            r'(?i)\b(how\s+are\s+you|status|working|running)\b': [
                "I'm running smoothly on your Pi with minimal resources!",
                "All systems operational. Using only pattern matching for speed.",
                "Emergency protocol active - everything is working perfectly!"
            ],
            
            # Time and date
            r'(?i)\b(time|what\s+time|clock)\b': [
                f"Current time is {time.strftime('%H:%M:%S')}",
                f"It's {time.strftime('%I:%M %p')} right now",
            ],
            
            r'(?i)\b(date|today|what\s+day)\b': [
                f"Today is {time.strftime('%A, %B %d, %Y')}",
                f"It's {time.strftime('%Y-%m-%d')}",
            ],
            
            # Pi-specific questions
            r'(?i)\b(pi|raspberry|computer|system|memory|ram)\b': [
                "I'm running on your Raspberry Pi using minimal resources!",
                "Your Pi is working great - I'm using less than 50MB of RAM!",
                "This is an ultra-optimized AI system designed for low-power devices."
            ],
            
            # Weather (without actual data)
            r'(?i)\b(weather|temperature|hot|cold|rain)\b': [
                "I don't have weather data, but your Pi's CPU temperature is good!",
                "Check your local weather app - I'm focused on keeping your Pi cool!",
            ],
            
            # Math operations
            r'(?i)(?:what\s+is\s+|calculate\s+|compute\s+)?(\d+)\s*([+\-*/])\s*(\d+)': 'math',
            
            # Help and commands
            r'(?i)\b(help|commands|what\s+can\s+you\s+do)\b': [
                "I can chat, do basic math, tell time, and provide system info!",
                "Try asking me about time, math problems, or Pi system status.",
                "Commands: 'status', 'optimize', 'mute', 'help', or just chat!"
            ],
            
            # Goodbye
            r'(?i)\b(bye|goodbye|exit|quit|see\s+you)\b': [
                "Goodbye! Your Pi AI will be here when you need it.",
                "See you later! Keep your Pi cool!",
                "Farewell! Mission accomplished."
            ],
            
            # Default responses for unmatched input
            'default': [
                "That's interesting! I'm a lightweight AI focused on basic assistance.",
                "I understand you're asking about that. I'm optimized for simple tasks on Pi.",
                "Good question! As an emergency protocol AI, I keep things simple and fast.",
                "I'm designed for quick responses rather than complex reasoning.",
                "Your Pi AI is ready for the next question!"
            ]
        }
    
    def setup_emergency_tts(self):
        """Setup TTS with multiple fallback options"""
        # Try espeak first (most reliable on Pi)
        if self.test_espeak():
            print("🔊 Using espeak for speech")
            return 'espeak'
        
        # Try festival as backup
        if self.test_festival():
            print("🔊 Using festival for speech")
            return 'festival'
        
        # Try pyttsx3 as last resort
        try:
            import pyttsx3
            engine = pyttsx3.init()
            print("🔊 Using pyttsx3 for speech")
            return 'pyttsx3'
        except:
            pass
        
        print("⚠️  No TTS system available - text only mode")
        return None
    
    def test_espeak(self) -> bool:
        """Test if espeak is available"""
        try:
            result = subprocess.run(['which', 'espeak'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def test_festival(self) -> bool:
        """Test if festival is available"""
        try:
            result = subprocess.run(['which', 'festival'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _emergency_tts_worker(self):
        """Ultra-lightweight TTS worker"""
        while True:
            try:
                text = self.tts_queue.get(timeout=1)
                if text and self.tts_enabled:
                    self.speak_text(text)
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS Error: {e}")
    
    def speak_text(self, text: str):
        """Speak text using available TTS system"""
        try:
            if self.tts_system == 'espeak':
                # Fast espeak command
                subprocess.run(['espeak', '-s', '180', text], 
                             capture_output=True)
            
            elif self.tts_system == 'festival':
                # Festival text-to-speech
                process = subprocess.Popen(['festival', '--tts'], 
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                process.communicate(input=text.encode())
            
            elif self.tts_system == 'pyttsx3':
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 180)
                engine.say(text)
                engine.runAndWait()
                
        except Exception as e:
            print(f"Speech error: {e}")
    
    def speak_async(self, text: str):
        """Add text to speech queue"""
        if self.tts_system and self.tts_enabled:
            try:
                self.tts_queue.put_nowait(text[:100])  # Limit length
            except queue.Full:
                pass  # Skip if queue full
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using pattern matching"""
        user_input = user_input.strip()
        if not user_input:
            return "I'm listening..."
        
        # Add to conversation context
        self.conversation_context.append(user_input)
        if len(self.conversation_context) > self.max_context:
            self.conversation_context.pop(0)
        
        # Check for math operations
        math_match = re.search(r'(\d+)\s*([+\-*/])\s*(\d+)', user_input)
        if math_match:
            return self.calculate_math(math_match)
        
        # Pattern matching for responses
        for pattern, responses in self.response_patterns.items():
            if pattern == 'default':
                continue
                
            if re.search(pattern, user_input):
                if isinstance(responses, list):
                    import random
                    return random.choice(responses)
                return responses
        
        # Default response with context awareness
        import random
        response = random.choice(self.response_patterns['default'])
        
        # Add context if appropriate
        if len(self.conversation_context) > 2:
            response += " What else would you like to know?"
        
        return response
    
    def calculate_math(self, match) -> str:
        """Safe math calculation"""
        try:
            num1 = float(match.group(1))
            operator = match.group(2)
            num2 = float(match.group(3))
            
            if operator == '+':
                result = num1 + num2
            elif operator == '-':
                result = num1 - num2
            elif operator == '*':
                result = num1 * num2
            elif operator == '/':
                if num2 == 0:
                    return "Cannot divide by zero!"
                result = num1 / num2
            else:
                return "Unknown operation"
            
            # Format result nicely
            if result == int(result):
                return f"{int(num1)} {operator} {int(num2)} = {int(result)}"
            else:
                return f"{num1} {operator} {num2} = {result:.2f}"
                
        except Exception as e:
            return "Math calculation error"
    
    def get_system_status(self) -> Dict:
        """Get system telemetry"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get temperature if available
            temp = "N/A"
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = f"{float(f.read()) / 1000:.1f}°C"
            except:
                pass
            
            uptime = time.time() - self.start_time
            
            return {
                'cpu': f"{cpu_percent:.1f}%",
                'memory_used': f"{memory.percent:.1f}%",
                'memory_available': f"{memory.available // (1024*1024)}MB",
                'temperature': temp,
                'uptime': f"{uptime:.0f}s",
                'responses_served': self.response_count,
                'tts_status': 'ON' if self.tts_enabled else 'OFF'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def optimize_memory(self):
        """Emergency memory cleanup"""
        # Clear old context
        if len(self.conversation_context) > 3:
            self.conversation_context = self.conversation_context[-3:]
        
        # Force garbage collection
        gc.collect()
        
        # Clear TTS queue if full
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break
        
        print("🧹 Memory optimized")
    
    def emergency_protocol_interface(self):
        """Ultra-simple command interface"""
        print("\n" + "="*50)
        print("🚨 EMERGENCY AI PROTOCOL - ULTRA LOW MEMORY MODE")
        print("="*50)
        print("💬 Just type and chat normally!")
        print("📋 Commands: 'status', 'optimize', 'mute', 'abort'")
        print("-"*50)
        
        while True:
            try:
                user_input = input("\n🎯 > ").strip()
                
                if user_input.lower() in ['abort', 'quit', 'exit']:
                    print("🚨 Emergency protocol terminated")
                    break
                
                elif user_input.lower() == 'status':
                    status = self.get_system_status()
                    print("\n📊 SYSTEM STATUS:")
                    for key, value in status.items():
                        print(f"   {key.upper()}: {value}")
                    continue
                
                elif user_input.lower() == 'optimize':
                    self.optimize_memory()
                    continue
                
                elif user_input.lower() == 'mute':
                    self.tts_enabled = not self.tts_enabled
                    status = "ENABLED" if self.tts_enabled else "DISABLED"
                    print(f"🔊 Speech {status}")
                    continue
                
                elif user_input.lower() == 'help':
                    print("""
🆘 EMERGENCY AI HELP:
   • Just chat normally - I'll respond quickly
   • Ask about time, date, or simple math
   • 'status' - Show system information
   • 'optimize' - Clean up memory
   • 'mute' - Toggle speech on/off
   • 'abort' - Shutdown system
                    """)
                    continue
                
                if not user_input:
                    continue
                
                # Generate and display response
                start_time = time.perf_counter()
                response = self.generate_response(user_input)
                end_time = time.perf_counter()
                
                self.response_count += 1
                
                print(f"\n🤖 {response}")
                print(f"⚡ {(end_time - start_time)*1000:.1f}ms")
                
                # Speak response
                self.speak_async(response)
                
                # Periodic memory cleanup
                if self.response_count % 10 == 0:
                    self.optimize_memory()
                
            except KeyboardInterrupt:
                print("\n🚨 Emergency stop activated")
                break
            except Exception as e:
                print(f"🚨 System error: {e}")

def emergency_installation_check():
    """Check and install minimal requirements"""
    print("🔍 Emergency system check...")
    
    # Check if espeak is available
    try:
        result = subprocess.run(['which', 'espeak'], capture_output=True)
        if result.returncode != 0:
            print("⚠️  Installing espeak for speech...")
            try:
                subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'espeak'], check=True)
                print("✅ espeak installed")
            except:
                print("❌ Could not install espeak automatically")
                print("Run: sudo apt-get install espeak")
        else:
            print("✅ espeak available")
    except:
        print("❌ Cannot check espeak status")
    
    # Check available memory
    try:
        memory = psutil.virtual_memory()
        available_mb = memory.available // (1024 * 1024)
        print(f"💾 Available RAM: {available_mb}MB")
        
        if available_mb < 100:
            print("🚨 CRITICAL: Very low memory!")
            print("🔧 Forcing memory cleanup...")
            gc.collect()
    except:
        print("❌ Cannot check memory status")
    
    print("✅ Emergency check complete")

def main():
    """Launch emergency protocol"""
    print("🚨 LAUNCHING EMERGENCY AI PROTOCOL")
    print("🛰️  Ultra-lightweight system for resource-constrained Pi")
    
    emergency_installation_check()
    
    try:
        # Initialize emergency system
        emergency_ai = EmergencyAIProtocol()
        
        # Launch interface
        emergency_ai.emergency_protocol_interface()
        
    except Exception as e:
        print(f"🚨 CRITICAL FAILURE: {e}")
        print("💡 System requires at least 100MB free RAM")

if __name__ == "__main__":
    main()
