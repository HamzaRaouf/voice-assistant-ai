import asyncio
import numpy as np
import soundfile as sf
import tempfile
import os
import logging
import threading
from io import BytesIO
from typing import Optional, Dict, Any, AsyncGenerator
from colorama import Fore
import re

# Import Kokoro ONNX - ensure this is installed
try:
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    logging.warning(f"{Fore.YELLOW}Kokoro ONNX not available. Install with: pip install kokoro-onnx{Fore.RESET}")

class LocalTTSEngine:
    """
    Local Text-to-Speech engine using Kokoro ONNX.
    Provides direct TTS generation without HTTP dependencies.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 voices_path: Optional[str] = None):
        """
        Initialize the local TTS engine.
        
        Args:
            model_path (str): Path to the Kokoro ONNX model file (None = use config)
            voices_path (str): Path to the voices binary file (None = use config)
        """
        # Import config here to avoid circular imports
        from .config import Config
        
        self.model_path = model_path or Config.KOKORO_MODEL_PATH
        self.voices_path = voices_path or Config.KOKORO_VOICES_PATH
        self.kokoro = None
        self._model_lock = threading.Lock()
        self.is_processing = False
        self.cancel_event = threading.Event()
        
        # Initialize model
        self._initialize_model()
        
        logging.info(f"{Fore.GREEN}LocalTTSEngine initialized with Kokoro model{Fore.RESET}")
    
    def _initialize_model(self):
        """Initialize the Kokoro model."""
        if not KOKORO_AVAILABLE:
            raise ImportError("Kokoro ONNX is not available. Please install it first.")
        
        try:
            with self._model_lock:
                if self.kokoro is None:
                    logging.info(f"{Fore.CYAN}Loading Kokoro TTS model: {self.model_path}{Fore.RESET}")
                    logging.info(f"{Fore.CYAN}Loading Kokoro voices: {self.voices_path}{Fore.RESET}")
                    
                    # Check if files exist
                    if not os.path.exists(self.model_path):
                        raise FileNotFoundError(f"Model file not found at {self.model_path}")
                    if not os.path.exists(self.voices_path):
                        raise FileNotFoundError(f"Voices file not found at {self.voices_path}")
                    
                    # Initialize Kokoro with the exact pattern you mentioned
                    self.kokoro = Kokoro(self.model_path, self.voices_path)
                    logging.info(f"{Fore.GREEN}Kokoro TTS model loaded successfully{Fore.RESET}")
        except Exception as e:
            logging.error(f"{Fore.RED}Failed to initialize Kokoro model: {e}{Fore.RESET}")
            raise
    
    def clean_and_format_text(self, input_text: str) -> str:
        """
        Clean and format text for better TTS output.
        
        Args:
            input_text (str): Raw input text
        
        Returns:
            str: Cleaned and formatted text
        """
        # Remove markdown formatting and special characters
        input_text = re.sub(r'^\s*(#\s*)+', '\n\g<0>', input_text)  # Adds line before heading
        input_text = re.sub(r'\n#', '\n\n#', input_text)
        input_text = re.sub(r'(\n|^)[\*\-\+] ', '\n\n• ', input_text)
        input_text = re.sub(r'(\n• [^•\n]+)(?=\n|$)', r'\1.', input_text)  # Adds period after bullet point text
        input_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', input_text)  # Removes '**' (bold)
        input_text = re.sub(r'\*([^*]+)\*', r'\1', input_text)  # Removes '*' (italic)
        input_text = re.sub(r'`([^`]+)`', r'\1', input_text)  # Removes '`' (code)
        input_text = input_text.replace("|", " ")  # If pipe character '|' is used as separator
        
        return input_text.strip()
    
    async def cancel_current_task(self):
        """Cancel the current TTS task."""
        if self.is_processing:
            logging.info(f"{Fore.YELLOW}Canceling current TTS task...{Fore.RESET}")
            self.cancel_event.set()
            return True
        return False
    
    def is_task_running(self) -> bool:
        """Check if a TTS task is currently running."""
        return self.is_processing
    
    async def generate_audio_stream(self, 
                                   text: str,
                                   voice: str = "af_jessica",
                                   speed: float = 0.9,
                                   lang: str = "en-us",
                                   cancellation_event: Optional[threading.Event] = None) -> AsyncGenerator[bytes, None]:
        """
        Generate audio as a stream of chunks using the exact Kokoro pattern.
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use (accent parameter)
            speed (float): Speech speed
            lang (str): Language code
            cancellation_event (threading.Event): External cancellation event
        
        Yields:
            bytes: Audio chunks
        """
        try:
            # Mark as processing and reset cancel event
            self.is_processing = True
            self.cancel_event.clear()
            
            # Ensure model is initialized
            if self.kokoro is None:
                self._initialize_model()
            
            # Clean the text
            formatted_text = self.clean_and_format_text(text)
            logging.info(f"{Fore.CYAN}Generating TTS stream for: '{formatted_text[:50]}...'{Fore.RESET}")
            
            # Create stream using the exact pattern you mentioned
            stream = self.kokoro.create_stream(formatted_text, voice=voice, speed=speed, lang=lang)
            
            # Process stream with cancellation support
            async for samples, sample_rate in stream:
                # Check for cancellation (internal or external)
                if (self.cancel_event.is_set() or 
                    (cancellation_event and cancellation_event.is_set())):
                    logging.info(f"{Fore.YELLOW}TTS stream canceled during processing{Fore.RESET}")
                    break
                
                # Convert samples to bytes (16-bit PCM format)
                # Ensure samples are in the correct format
                if len(samples) > 0:
                    # Convert to 16-bit PCM
                    samples_int16 = (samples * 32767).astype(np.int16)
                    audio_bytes = samples_int16.tobytes()
                    yield audio_bytes
                
                # Allow other async tasks to run
                await asyncio.sleep(0.001)
            
            logging.info(f"{Fore.GREEN}TTS stream generation completed{Fore.RESET}")
            
        except Exception as e:
            logging.error(f"{Fore.RED}Error generating TTS stream: {e}{Fore.RESET}")
            raise
        finally:
            self.is_processing = False
    
    async def generate_audio_file(self, 
                                 text: str,
                                 output_path: str,
                                 voice: str = "af_jessica",
                                 speed: float = 0.9,
                                 lang: str = "en-us",
                                 cancellation_event: Optional[threading.Event] = None) -> Optional[str]:
        """
        Generate audio and save to file.
        
        Args:
            text (str): Text to convert to speech
            output_path (str): Path to save the audio file
            voice (str): Voice to use
            speed (float): Speech speed
            lang (str): Language code
            cancellation_event (threading.Event): Cancellation event
        
        Returns:
            str: Path to the generated audio file, or None if cancelled
        """
        try:
            # Check for cancellation before starting
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}TTS file generation cancelled before starting{Fore.RESET}")
                return None
            
            logging.info(f"{Fore.CYAN}Generating TTS file: {output_path}{Fore.RESET}")
            
            # Collect all audio chunks
            all_samples = []
            sample_rate = 24000  # Default sample rate for Kokoro
            
            async for audio_chunk in self.generate_audio_stream(
                text=text,
                voice=voice,
                speed=speed,
                lang=lang,
                cancellation_event=cancellation_event
            ):
                # Check for cancellation
                if cancellation_event and cancellation_event.is_set():
                    logging.info(f"{Fore.YELLOW}TTS file generation cancelled during stream collection{Fore.RESET}")
                    return None
                
                # Convert bytes back to samples for accumulation
                samples_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
                samples_float = samples_int16.astype(np.float32) / 32767.0
                all_samples.append(samples_float)
            
            # Check for cancellation before final processing
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}TTS file generation cancelled before saving{Fore.RESET}")
                return None
            
            if all_samples:
                # Concatenate all samples
                combined_samples = np.concatenate(all_samples)
                
                # Save to file using soundfile
                sf.write(output_path, combined_samples, sample_rate, format='WAV')
                logging.info(f"{Fore.GREEN}TTS audio saved to: {output_path}{Fore.RESET}")
                return output_path
            else:
                logging.warning(f"{Fore.YELLOW}No audio samples generated{Fore.RESET}")
                return None
            
        except Exception as e:
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}TTS file generation cancelled due to interruption{Fore.RESET}")
                return None
            logging.error(f"{Fore.RED}Error generating TTS file: {e}{Fore.RESET}")
            raise
    
    async def generate_audio_buffer(self, 
                                   text: str,
                                   voice: str = "af_jessica",
                                   speed: float = 0.9,
                                   lang: str = "en-us",
                                   cancellation_event: Optional[threading.Event] = None) -> Optional[BytesIO]:
        """
        Generate audio and return as BytesIO buffer.
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use
            speed (float): Speech speed
            lang (str): Language code
            cancellation_event (threading.Event): Cancellation event
        
        Returns:
            BytesIO: Audio buffer in WAV format, or None if cancelled
        """
        try:
            # Check for cancellation before starting
            if cancellation_event and cancellation_event.is_set():
                return None
            
            logging.info(f"{Fore.CYAN}Generating TTS buffer{Fore.RESET}")
            
            # Create an in-memory buffer
            buffer = BytesIO()
            
            # Collect all audio chunks
            all_samples = []
            sample_rate = 24000  # Default sample rate for Kokoro
            
            async for audio_chunk in self.generate_audio_stream(
                text=text,
                voice=voice,
                speed=speed,
                lang=lang,
                cancellation_event=cancellation_event
            ):
                # Check for cancellation
                if cancellation_event and cancellation_event.is_set():
                    return None
                
                # Convert bytes back to samples for accumulation
                samples_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
                samples_float = samples_int16.astype(np.float32) / 32767.0
                all_samples.append(samples_float)
            
            # Check for cancellation before final processing
            if cancellation_event and cancellation_event.is_set():
                return None
            
            if all_samples:
                # Concatenate all samples
                combined_samples = np.concatenate(all_samples)
                
                # Write to buffer as WAV
                sf.write(buffer, combined_samples, sample_rate, format='WAV')
                buffer.seek(0)
                
                logging.info(f"{Fore.GREEN}TTS audio buffer created successfully{Fore.RESET}")
                return buffer
            else:
                logging.warning(f"{Fore.YELLOW}No audio samples generated for buffer{Fore.RESET}")
                return None
            
        except Exception as e:
            if cancellation_event and cancellation_event.is_set():
                return None
            logging.error(f"{Fore.RED}Error generating TTS buffer: {e}{Fore.RESET}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self.model_path,
            "voices_path": self.voices_path,
            "is_processing": self.is_processing,
            "kokoro_available": KOKORO_AVAILABLE
        }

# Global instance for easy access
_global_tts_engine = None

def get_local_tts_engine(model_path: Optional[str] = None,
                        voices_path: Optional[str] = None) -> LocalTTSEngine:
    """
    Get or create a global TTS engine instance.
    
    Args:
        model_path (str): Path to the Kokoro ONNX model file (None = use config)
        voices_path (str): Path to the voices binary file (None = use config)
    
    Returns:
        LocalTTSEngine: The TTS engine instance
    """
    global _global_tts_engine
    
    if _global_tts_engine is None:
        _global_tts_engine = LocalTTSEngine(
            model_path=model_path,
            voices_path=voices_path
        )
    
    return _global_tts_engine

async def generate_speech_local(text: str,
                               output_path: Optional[str] = None,
                               voice: str = "af_jessica",
                               speed: float = 0.9,
                               lang: str = "en-us",
                               cancellation_event: Optional[threading.Event] = None) -> Optional[str]:
    """
    Convenience function for simple TTS generation.
    
    Args:
        text (str): Text to convert to speech
        output_path (str): Path to save audio file (if None, creates temp file)
        voice (str): Voice to use
        speed (float): Speech speed
        lang (str): Language code
        cancellation_event (threading.Event): Cancellation event
    
    Returns:
        str: Path to generated audio file or None if cancelled/failed
    """
    try:
        engine = get_local_tts_engine()
        
        # Create output path if not provided
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                output_path = temp_file.name
        
        result = await engine.generate_audio_file(
            text=text,
            output_path=output_path,
            voice=voice,
            speed=speed,
            lang=lang,
            cancellation_event=cancellation_event
        )
        
        return result
        
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            return None
        logging.error(f"{Fore.RED}Local TTS convenience function error: {e}{Fore.RESET}")
        return None

async def generate_speech_stream_local(text: str,
                                      voice: str = "af_jessica",
                                      speed: float = 0.9,
                                      lang: str = "en-us",
                                      cancellation_event: Optional[threading.Event] = None) -> AsyncGenerator[bytes, None]:
    """
    Convenience function for streaming TTS generation.
    
    Args:
        text (str): Text to convert to speech
        voice (str): Voice to use
        speed (float): Speech speed
        lang (str): Language code
        cancellation_event (threading.Event): Cancellation event
    
    Yields:
        bytes: Audio chunks
    """
    try:
        engine = get_local_tts_engine()
        
        async for chunk in engine.generate_audio_stream(
            text=text,
            voice=voice,
            speed=speed,
            lang=lang,
            cancellation_event=cancellation_event
        ):
            yield chunk
            
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            return
        logging.error(f"{Fore.RED}Local TTS streaming convenience function error: {e}{Fore.RESET}")
        return 