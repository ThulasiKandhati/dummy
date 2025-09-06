#!/usr/bin/env python3
"""
Complete Voice Assistant - Transcription -> LLM -> TTS Pipeline
Optimized for speed with all components in one file
"""

import speech_recognition as sr
import time
import os
import sys
import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSEngine:
    """Text-to-Speech Engine using eSpeak-NG"""
    
    def __init__(self, voice: str = "en", speed: int = 200, pitch: int = 45):
        self.voice = voice
        self.speed = speed
        self.pitch = pitch
        self.output_dir = Path("/app/output")
        self.output_dir.mkdir(exist_ok=True)
    
    def speak(self, text: str, output_file: Optional[str] = None) -> bool:
        """Convert text to speech"""
        try:
            cmd = [
                "espeak-ng",
                "-v", self.voice,
                "-s", str(self.speed),
                "-p", str(self.pitch),
                text
            ]
            
            if output_file:
                output_path = self.output_dir / output_file
                cmd.extend(["-w", str(output_path)])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
                
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            return False

class LLMEngine:
    """Google Generative AI LLM Engine"""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Google Generative AI model"""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.warning("GOOGLE_API_KEY not set - LLM disabled")
                return
            
            genai.configure(api_key=api_key)
            
            # Try models in order of preference
            model_names = [
                'gemini-1.5-flash',
                'gemini-1.5-pro', 
                'models/gemini-1.5-flash',
                'gemini-pro'
            ]
            
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    # Quick test
                    test_response = model.generate_content("Hi")
                    self.model = model
                    logger.info(f"✅ LLM initialized: {model_name}")
                    return
                except Exception:
                    continue
            
            logger.error("❌ All LLM models failed")
            
        except ImportError:
            logger.warning("google-generativeai not installed")
        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response from the model"""
        if not self.model:
            return None
            
        try:
            import google.generativeai as genai
            
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=150,  # Keep responses short for speed
            )
            
            response = self.model.generate_content(prompt, generation_config=generation_config)
            
            if response.candidates and response.candidates[0].content:
                return response.candidates[0].content.parts[0].text
            else:
                return None
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None

class VoiceAssistant:
    """Complete Voice Assistant Pipeline"""
    
    def __init__(self, whisper_model="base"):
        """Initialize all components"""
        print("🚀 Initializing Voice Assistant...")
        
        # Initialize Whisper
        print(f"🔄 Loading Whisper model ({whisper_model})...")
        self.whisper_model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        print("✅ Whisper loaded!")
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = self._setup_microphone()
        
        # Initialize LLM
        print("🧠 Loading LLM...")
        self.llm = LLMEngine()
        
        # Initialize TTS
        print("🔊 Loading TTS...")
        self.tts = TTSEngine()
        
        # Performance settings
        self.is_listening = True
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        print("✅ Voice Assistant ready!")
    
    def _setup_microphone(self):
        """Setup and calibrate microphone"""
        print("🎤 Setting up microphone...")
        try:
            microphone = sr.Microphone()
            
            # Quick calibration
            print("🔧 Calibrating... (1 second)")
            with microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("✅ Microphone ready!")
            return microphone
            
        except Exception as e:
            print(f"❌ Microphone error: {e}")
            raise
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Whisper"""
        try:
            # Save to temp file
            temp_file = "temp_audio.wav"
            with open(temp_file, "wb") as f:
                f.write(audio_data.get_wav_data())
            
            # Transcribe
            segments, info = self.whisper_model.transcribe(temp_file, beam_size=5)
            
            transcription = ""
            for segment in segments:
                transcription += segment.text + " "
            
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return transcription.strip(), info
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", None
    
    def process_voice_input(self, text: str, language_info=None):
        """Main pipeline: Text -> LLM -> TTS"""
        start_time = time.time()
        
        print(f"💬 You: '{text}'")
        
        # Handle exit commands
        if any(cmd in text.lower() for cmd in ["stop", "quit", "exit", "goodbye"]):
            response = "Goodbye!"
            print(f"🤖 Assistant: {response}")
            self.tts.speak(response)
            self.is_listening = False
            return
        
        # Quick responses for common queries
        text_lower = text.lower()
        if "hello" in text_lower or "hi" in text_lower:
            response = "Hello! How can I help you?"
        elif "time" in text_lower:
            response = time.strftime("It's %I:%M %p")
        elif "date" in text_lower:
            response = time.strftime("Today is %A, %B %d")
        else:
            # Get LLM response
            if self.llm.model:
                prompt = f"You are a helpful voice assistant. Respond naturally in 1-2 sentences to: '{text}'"
                response = self.llm.generate_response(prompt)
                if not response:
                    response = "I'm sorry, I couldn't generate a response right now."
            else:
                response = f"I heard you say: {text}"
        
        # Output response
        print(f"🤖 Assistant: {response}")
        
        # Speak response (non-blocking)
        self.executor.submit(self.tts.speak, response)
        
        # Performance timing
        total_time = time.time() - start_time
        print(f"⚡ Pipeline completed in {total_time:.3f}s")
    
    def start_listening(self):
        """Start the main listening loop"""
        print("\n" + "="*60)
        print("🎙️ VOICE ASSISTANT ACTIVE")
        print("="*60)
        print("💡 Speak clearly into your microphone")
        print("💡 Say 'stop', 'quit', or 'exit' to end")
        print("🧠 LLM:", "✅ READY" if self.llm.model else "❌ DISABLED")
        print("🔊 TTS: ✅ READY")
        print("="*60)
        
        while self.is_listening:
            try:
                print("\n🎤 Listening...")
                
                # Listen for audio
                with self.microphone as source:
                    self.recognizer.energy_threshold = 300
                    self.recognizer.dynamic_energy_threshold = True
                    
                    audio = self.recognizer.listen(
                        source, 
                        timeout=1,
                        phrase_time_limit=8
                    )
                
                print("🔄 Transcribing...")
                
                # Transcribe
                transcription, info = self.transcribe_audio(audio)
                
                if transcription:
                    # Process through pipeline
                    self.process_voice_input(transcription, info)
                else:
                    print("❓ No speech detected")
                
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("❓ Could not understand audio")
                continue
            except KeyboardInterrupt:
                print("\n🛑 Interrupted")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(1)
                continue
        
        print("\n👋 Voice Assistant stopped")
        self.executor.shutdown(wait=True)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="High-Speed Voice Assistant")
    parser.add_argument("--model", default="base", 
                       help="Whisper model size (tiny/base/small/medium)")
    parser.add_argument("--test-tts", help="Test TTS with given text")
    parser.add_argument("--test-llm", help="Test LLM with given prompt")
    
    args = parser.parse_args()
    
    # Test modes
    if args.test_tts:
        print("🔊 Testing TTS...")
        tts = TTSEngine()
        success = tts.speak(args.test_tts)
        print("✅ TTS test completed" if success else "❌ TTS test failed")
        return
    
    if args.test_llm:
        print("🧠 Testing LLM...")
        llm = LLMEngine()
        if llm.model:
            response = llm.generate_response(args.test_llm)
            print(f"Response: {response}")
        else:
            print("❌ LLM not available")
        return
    
    # Main voice assistant
    try:
        assistant = VoiceAssistant(whisper_model=args.model)
        assistant.start_listening()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        print("💡 Make sure microphone is connected and dependencies are installed")

if __name__ == "__main__":
    main()
