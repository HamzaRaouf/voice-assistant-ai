import os
import logging
import asyncio
import concurrent.futures
from typing import Optional, Dict, Any, List
from faster_whisper import WhisperModel
from colorama import Fore
import threading
import numpy as np
import librosa
from io import BytesIO
import pydub
from .config import Config

# Set environment variable to avoid library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class LocalSTTEngine:
    """
    Local Speech-to-Text engine using faster-whisper.
    Provides direct transcription without HTTP dependencies.
    """
    
    def __init__(self, 
                 model_size: str = None, 
                 device: str = "auto", 
                 compute_type: str = "int8",
                 max_workers: int = 6):
        """
        Initialize the local STT engine.
        
        Args:
            model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large'). If None, uses Config.WHISPER_MODEL_SIZE
            device (str): Device to use ('cpu', 'cuda', 'auto')
            compute_type (str): Compute type ('int8', 'float16', 'float32')
            max_workers (int): Maximum number of worker threads
        """
        self.model_size = model_size or Config.WHISPER_MODEL_SIZE
        self.device = device if device != "auto" else self._detect_best_device()
        self.compute_type = compute_type if self.device == "cuda" else "int8"
        self.max_workers = max_workers
        self.model = None
        self._model_lock = threading.Lock()
        
        # Initialize model
        self._initialize_model()
        
        logging.info(f"{Fore.GREEN}LocalSTTEngine initialized: {model_size} model on {self.device} with {compute_type}{Fore.RESET}")
    
    def _detect_best_device(self) -> str:
        """Detect the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    
    def _initialize_model(self):
        """Initialize the Whisper model."""
        try:
            with self._model_lock:
                if self.model is None:
                    logging.info(f"{Fore.CYAN}Loading Whisper model: {self.model_size} on {self.device}{Fore.RESET}")
                    self.model = WhisperModel(
                        self.model_size, 
                        device=self.device, 
                        compute_type=self.compute_type
                    )
                    logging.info(f"{Fore.GREEN}Whisper model loaded successfully{Fore.RESET}")
        except Exception as e:
            logging.error(f"{Fore.RED}Failed to initialize Whisper model: {e}{Fore.RESET}")
            raise
    
    def _preprocess_audio_data(self, audio_data: np.ndarray, original_rate: int = 22050, target_rate: int = 16000) -> np.ndarray:
        """
        Preprocess audio data similar to stt_server.py for optimal Whisper performance.
        
        Args:
            audio_data (np.ndarray): Raw audio data
            original_rate (int): Original sample rate
            target_rate (int): Target sample rate for Whisper
            
        Returns:
            np.ndarray: Preprocessed audio data
        """
        try:
            # Convert to float64 for high precision processing
            audio_data = audio_data.astype(np.float64)
            
            logging.info(f"🔄 Processing audio: {original_rate} Hz → {target_rate} Hz")
            
            # High-quality resampling if needed
            if original_rate != target_rate:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=original_rate, 
                    target_sr=target_rate,
                    res_type='kaiser_best'
                )
            
            # Normalize and enhance audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.85
            
            # Apply pre-emphasis filter
            pre_emphasis = np.float64(0.95)
            audio_data = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
            
            # Apply noise gate
            noise_threshold = np.float64(0.005)
            audio_data = np.where(np.abs(audio_data) > noise_threshold, audio_data, np.float64(0))
            
            # Ensure correct dtype for Whisper
            audio_data = audio_data.astype(np.float32)
            
            return audio_data
            
        except Exception as e:
            logging.error(f"{Fore.RED}Audio preprocessing error: {e}{Fore.RESET}")
            raise
    
    def transcribe_audio_buffer(self, 
                               audio_buffer: BytesIO,
                               language: Optional[str] = "en",
                               initial_prompt: Optional[str] = None,
                               word_timestamps: bool = False,
                               vad_filter: bool = True,
                               min_silence_duration_ms: int = 500,  # Reduced default for less aggressive VAD
                               temperature: float = 0.0,
                               best_of: int = 5,
                               beam_size: int = 5,
                               patience: float = 1.0,
                               length_penalty: float = 1.0,
                               compression_ratio_threshold: float = 2.4,
                               no_speech_threshold: float = 0.5,  # More sensitive to speech
                               cancellation_event: Optional[threading.Event] = None) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio data directly from memory buffer without saving to file.
        
        Args:
            audio_buffer (BytesIO): Audio data in memory (MP3 format)
            language (str): Language code (e.g., 'en' for English). None for auto-detection
            initial_prompt (str): Initial prompt to guide transcription
            word_timestamps (bool): Whether to include word-level timestamps
            vad_filter (bool): Whether to apply voice activity detection
            min_silence_duration_ms (int): Minimum silence duration in ms
            temperature (float): Temperature for sampling (0.0 = deterministic)
            best_of (int): Number of candidates to generate
            beam_size (int): Beam size for beam search
            patience (float): Patience parameter for beam search
            length_penalty (float): Length penalty for beam search
            compression_ratio_threshold (float): Compression ratio threshold
            no_speech_threshold (float): No speech threshold
            cancellation_event (threading.Event): Event to check for cancellation
        
        Returns:
            dict: Transcription result with text and optional timestamps, or None if cancelled
        """
        try:
            # Check for cancellation before starting
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}Local STT transcription cancelled before starting{Fore.RESET}")
                return None
            
            # Ensure model is initialized
            if self.model is None:
                self._initialize_model()
            
            logging.info(f"{Fore.CYAN}Starting direct audio transcription from memory{Fore.RESET}")
            
            # Convert audio buffer to numpy array
            audio_buffer.seek(0)  # Reset to beginning
            audio_segment = pydub.AudioSegment.from_mp3(audio_buffer)
            
            # Get audio data and sample rate
            audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            original_rate = audio_segment.frame_rate
            
            # Convert to mono if stereo
            if audio_segment.channels == 2:
                audio_data = audio_data.reshape((-1, 2))
                audio_data = audio_data.mean(axis=1)
            
            # Normalize to [-1, 1] range
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Preprocess audio for optimal Whisper performance
            audio_data = self._preprocess_audio_data(audio_data, original_rate, target_rate=16000)
            
            # Check for cancellation before transcription
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}Local STT transcription cancelled before model call{Fore.RESET}")
                return None
            
            # Configure VAD parameters
            vad_parameters = None
            if vad_filter:
                vad_parameters = {
                    "min_silence_duration_ms": min_silence_duration_ms,
                    "threshold": 0.3  # Lower threshold to be less aggressive
                }
            
            logging.info("🎯 Transcribing with Whisper...")
            
            # Perform transcription with enhanced parameters
            segments, info = self.model.transcribe(
                audio_data,
                language=language,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                temperature=temperature,
                best_of=best_of,
                beam_size=beam_size,
                patience=patience,
                length_penalty=length_penalty,
                compression_ratio_threshold=compression_ratio_threshold,
                no_speech_threshold=no_speech_threshold
            )
            
            # Check for cancellation after model call
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}Local STT transcription cancelled after model call{Fore.RESET}")
                return None
            
            # Extract transcription text and segments
            transcription_text = ""
            segments_data = []
            
            for segment in segments:
                # Check for cancellation during segment processing
                if cancellation_event and cancellation_event.is_set():
                    logging.info(f"{Fore.YELLOW}Local STT transcription cancelled during segment processing{Fore.RESET}")
                    return None
                
                transcription_text += segment.text
                
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
                
                # Add word-level timestamps if requested
                if word_timestamps and hasattr(segment, 'words'):
                    segment_data["words"] = []
                    for word in segment.words:
                        word_data = {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word
                        }
                        segment_data["words"].append(word_data)
                
                segments_data.append(segment_data)
            
            # If VAD removed all audio but we have duration, try without VAD
            if not transcription_text.strip() and info.duration > 0.5 and vad_filter:
                logging.warning(f"{Fore.YELLOW}VAD removed all audio, retrying without VAD filter{Fore.RESET}")
                
                # Retry without VAD filter
                segments_retry, info_retry = self.model.transcribe(
                    audio_data,
                    language=language,
                    initial_prompt=initial_prompt,
                    word_timestamps=word_timestamps,
                    vad_filter=False,  # Disable VAD
                    vad_parameters=None,
                    temperature=temperature,
                    best_of=best_of,
                    beam_size=beam_size,
                    patience=patience,
                    length_penalty=length_penalty,
                    compression_ratio_threshold=compression_ratio_threshold,
                    no_speech_threshold=no_speech_threshold
                )
                
                # Extract transcription from retry
                transcription_text_retry = ""
                segments_data_retry = []
                
                for segment in segments_retry:
                    if cancellation_event and cancellation_event.is_set():
                        return None
                    
                    transcription_text_retry += segment.text
                    
                    segment_data = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    }
                    
                    if word_timestamps and hasattr(segment, 'words'):
                        segment_data["words"] = []
                        for word in segment.words:
                            word_data = {
                                "start": word.start,
                                "end": word.end,
                                "word": word.word
                            }
                            segment_data["words"].append(word_data)
                    
                    segments_data_retry.append(segment_data)
                
                # Use retry results if they contain text
                if transcription_text_retry.strip():
                    logging.info(f"{Fore.GREEN}Retry without VAD successful{Fore.RESET}")
                    transcription_text = transcription_text_retry
                    segments_data = segments_data_retry
                    info = info_retry
            
            # Final cancellation check
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}Local STT transcription cancelled after processing{Fore.RESET}")
                return None
            
            # Prepare result
            result = {
                "text": transcription_text.strip(),
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": segments_data
            }
            
            logging.info(f"{Fore.GREEN}📋 Transcription: '{transcription_text.strip()}'{Fore.RESET}")
            return result
            
        except Exception as e:
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}Local STT transcription cancelled due to interruption{Fore.RESET}")
                return None
            logging.error(f"{Fore.RED}Local STT transcription error: {e}{Fore.RESET}")
            raise
    
    def transcribe(self, 
                   audio_file_path: str,
                   language: Optional[str] = "en",
                   initial_prompt: Optional[str] = None,
                   word_timestamps: bool = False,
                   vad_filter: bool = True,
                   min_silence_duration_ms: int = 1000,
                   cancellation_event: Optional[threading.Event] = None) -> Optional[Dict[str, Any]]:
        """
        Transcribe an audio file directly using faster-whisper.
        
        Args:
            audio_file_path (str): Path to the audio file
            language (str): Language code (e.g., 'en' for English). None for auto-detection
            initial_prompt (str): Initial prompt to guide transcription
            word_timestamps (bool): Whether to include word-level timestamps
            vad_filter (bool): Whether to apply voice activity detection
            min_silence_duration_ms (int): Minimum silence duration in ms
            cancellation_event (threading.Event): Event to check for cancellation
        
        Returns:
            dict: Transcription result with text and optional timestamps, or None if cancelled
        """
        try:
            # Check for cancellation before starting
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}Local STT transcription cancelled before starting{Fore.RESET}")
                return None
            
            # Ensure model is initialized
            if self.model is None:
                self._initialize_model()
            
            # Check if file exists
            if not os.path.exists(audio_file_path):
                logging.error(f"{Fore.RED}Audio file not found: {audio_file_path}{Fore.RESET}")
                return None
            
            logging.info(f"{Fore.CYAN}Starting local transcription: {audio_file_path}{Fore.RESET}")
            
            # Configure VAD parameters
            vad_parameters = None
            if vad_filter:
                vad_parameters = {
                    "min_silence_duration_ms": min_silence_duration_ms,
                    "threshold": 0.3  # Lower threshold to be less aggressive
                }
            
            # Check for cancellation before transcription
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}Local STT transcription cancelled before model call{Fore.RESET}")
                return None
            
            # Perform transcription with VAD
            segments, info = self.model.transcribe(
                audio_file_path,
                language=language,
                initial_prompt=initial_prompt,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters
            )
            
            # Check for cancellation after model call
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}Local STT transcription cancelled after model call{Fore.RESET}")
                return None
            
            # Extract transcription text and segments
            transcription_text = ""
            segments_data = []
            
            for segment in segments:
                # Check for cancellation during segment processing
                if cancellation_event and cancellation_event.is_set():
                    logging.info(f"{Fore.YELLOW}Local STT transcription cancelled during segment processing{Fore.RESET}")
                    return None
                
                transcription_text += segment.text
                
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text
                }
                
                # Add word-level timestamps if requested
                if word_timestamps and hasattr(segment, 'words'):
                    segment_data["words"] = []
                    for word in segment.words:
                        word_data = {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word
                        }
                        segment_data["words"].append(word_data)
                
                segments_data.append(segment_data)
            
            # If VAD removed all audio but we have duration, try without VAD
            if not transcription_text.strip() and info.duration > 0.5 and vad_filter:
                logging.warning(f"{Fore.YELLOW}VAD removed all audio, retrying without VAD filter{Fore.RESET}")
                
                # Retry without VAD filter
                segments_retry, info_retry = self.model.transcribe(
                    audio_file_path,
                    language=language,
                    initial_prompt=initial_prompt,
                    word_timestamps=word_timestamps,
                    vad_filter=False,  # Disable VAD
                    vad_parameters=None
                )
                
                # Extract transcription from retry
                transcription_text_retry = ""
                segments_data_retry = []
                
                for segment in segments_retry:
                    if cancellation_event and cancellation_event.is_set():
                        return None
                    
                    transcription_text_retry += segment.text
                    
                    segment_data = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    }
                    
                    if word_timestamps and hasattr(segment, 'words'):
                        segment_data["words"] = []
                        for word in segment.words:
                            word_data = {
                                "start": word.start,
                                "end": word.end,
                                "word": word.word
                            }
                            segment_data["words"].append(word_data)
                    
                    segments_data_retry.append(segment_data)
                
                # Use retry results if they contain text
                if transcription_text_retry.strip():
                    logging.info(f"{Fore.GREEN}Retry without VAD successful{Fore.RESET}")
                    transcription_text = transcription_text_retry
                    segments_data = segments_data_retry
                    info = info_retry
            
            # Final cancellation check
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}Local STT transcription cancelled after processing{Fore.RESET}")
                return None
            
            # Prepare result
            result = {
                "text": transcription_text.strip(),
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": segments_data
            }
            
            logging.info(f"{Fore.GREEN}Local transcription completed: '{transcription_text.strip()[:100]}...'{Fore.RESET}")
            return result
            
        except Exception as e:
            if cancellation_event and cancellation_event.is_set():
                logging.info(f"{Fore.YELLOW}Local STT transcription cancelled due to interruption{Fore.RESET}")
                return None
            logging.error(f"{Fore.RED}Local STT transcription error: {e}{Fore.RESET}")
            raise
    
    async def transcribe_async(self, 
                               audio_file_path: str,
                               language: Optional[str] = "en",
                               initial_prompt: Optional[str] = None,
                               word_timestamps: bool = False,
                               vad_filter: bool = True,
                               min_silence_duration_ms: int = 1000,
                               cancellation_event: Optional[threading.Event] = None) -> Optional[Dict[str, Any]]:
        """
        Async wrapper for transcription using thread pool.
        
        Args:
            Same as transcribe method
        
        Returns:
            dict: Transcription result or None if cancelled
        """
        try:
            # Check for cancellation before starting
            if cancellation_event and cancellation_event.is_set():
                return None
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                result = await loop.run_in_executor(
                    executor,
                    self.transcribe,
                    audio_file_path,
                    language,
                    initial_prompt,
                    word_timestamps,
                    vad_filter,
                    min_silence_duration_ms,
                    cancellation_event
                )
            
            return result
            
        except Exception as e:
            if cancellation_event and cancellation_event.is_set():
                return None
            logging.error(f"{Fore.RED}Async local STT transcription error: {e}{Fore.RESET}")
            raise
    
    async def transcribe_audio_buffer_async(self, 
                                          audio_buffer: BytesIO,
                                          language: Optional[str] = "en",
                                          initial_prompt: Optional[str] = None,
                                          word_timestamps: bool = False,
                                          vad_filter: bool = True,
                                          min_silence_duration_ms: int = 500,  # Reduced default for less aggressive VAD
                                          temperature: float = 0.0,
                                          best_of: int = 5,
                                          beam_size: int = 5,
                                          patience: float = 1.0,
                                          length_penalty: float = 1.0,
                                          compression_ratio_threshold: float = 2.4,
                                          no_speech_threshold: float = 0.5,  # More sensitive to speech
                                          cancellation_event: Optional[threading.Event] = None) -> Optional[Dict[str, Any]]:
        """
        Async wrapper for buffer transcription using thread pool.
        
        Args:
            Same as transcribe_audio_buffer method
        
        Returns:
            dict: Transcription result or None if cancelled
        """
        try:
            # Check for cancellation before starting
            if cancellation_event and cancellation_event.is_set():
                return None
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                result = await loop.run_in_executor(
                    executor,
                    self.transcribe_audio_buffer,
                    audio_buffer,
                    language,
                    initial_prompt,
                    word_timestamps,
                    vad_filter,
                    min_silence_duration_ms,
                    temperature,
                    best_of,
                    beam_size,
                    patience,
                    length_penalty,
                    compression_ratio_threshold,
                    no_speech_threshold,
                    cancellation_event
                )
            
            return result
            
        except Exception as e:
            if cancellation_event and cancellation_event.is_set():
                return None
            logging.error(f"{Fore.RED}Async local STT buffer transcription error: {e}{Fore.RESET}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "max_workers": self.max_workers
        }

# Global instance for easy access
_global_stt_engine = None

def get_local_stt_engine(model_size: str = None, 
                        device: str = "auto", 
                        compute_type: str = "int8") -> LocalSTTEngine:
    """
    Get or create a global STT engine instance.
    
    Args:
        model_size (str): Whisper model size. If None, uses Config.WHISPER_MODEL_SIZE
        device (str): Device to use
        compute_type (str): Compute type
    
    Returns:
        LocalSTTEngine: The STT engine instance
    """
    global _global_stt_engine
    
    if _global_stt_engine is None:
        _global_stt_engine = LocalSTTEngine(
            model_size=model_size or Config.WHISPER_MODEL_SIZE,
            device=device,
            compute_type=compute_type
        )
    
    return _global_stt_engine

def transcribe_audio_local(audio_file_path: str,
                          model_size: str = None,
                          language: Optional[str] = "en",
                          initial_prompt: Optional[str] = None,
                          word_timestamps: bool = False,
                          vad_filter: bool = True,
                          min_silence_duration_ms: int = 1000,
                          cancellation_event: Optional[threading.Event] = None) -> Optional[str]:
    """
    Convenience function for simple text transcription.
    
    Args:
        audio_file_path (str): Path to the audio file
        model_size (str): Whisper model size. If None, uses Config.WHISPER_MODEL_SIZE
        language (str): Language code
        initial_prompt (str): Initial prompt
        word_timestamps (bool): Include word timestamps
        vad_filter (bool): Apply VAD filter
        min_silence_duration_ms (int): Minimum silence duration
        cancellation_event (threading.Event): Cancellation event
    
    Returns:
        str: Transcribed text or None if cancelled/failed
    """
    try:
        engine = get_local_stt_engine(model_size=model_size or Config.WHISPER_MODEL_SIZE)
        result = engine.transcribe(
            audio_file_path=audio_file_path,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            min_silence_duration_ms=min_silence_duration_ms,
            cancellation_event=cancellation_event
        )
        
        if result:
            return result.get("text", "")
        return None
        
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            return None
        logging.error(f"{Fore.RED}Local STT convenience function error: {e}{Fore.RESET}")
        return None

def transcribe_audio_buffer_local(audio_buffer: BytesIO,
                                 model_size: str = None,
                                 language: Optional[str] = "en",
                                 initial_prompt: Optional[str] = None,
                                 word_timestamps: bool = False,
                                 vad_filter: bool = True,
                                 min_silence_duration_ms: int = 500,  # Less aggressive VAD default
                                 temperature: float = 0.0,
                                 best_of: int = 5,
                                 beam_size: int = 5,
                                 patience: float = 1.0,
                                 length_penalty: float = 1.0,
                                 compression_ratio_threshold: float = 2.4,
                                 no_speech_threshold: float = 0.5,  # More sensitive to speech
                                 cancellation_event: Optional[threading.Event] = None) -> Optional[str]:
    """
    Convenience function for simple text transcription from audio buffer.
    
    Args:
        audio_buffer (BytesIO): Audio data in memory (MP3 format)
        model_size (str): Whisper model size. If None, uses Config.WHISPER_MODEL_SIZE
        language (str): Language code
        initial_prompt (str): Initial prompt
        word_timestamps (bool): Include word timestamps
        vad_filter (bool): Apply VAD filter
        min_silence_duration_ms (int): Minimum silence duration
        temperature (float): Temperature for sampling
        best_of (int): Number of candidates to generate
        beam_size (int): Beam size for beam search
        patience (float): Patience parameter for beam search
        length_penalty (float): Length penalty for beam search
        compression_ratio_threshold (float): Compression ratio threshold
        no_speech_threshold (float): No speech threshold
        cancellation_event (threading.Event): Cancellation event
    
    Returns:
        str: Transcribed text or None if cancelled/failed
    """
    try:
        engine = get_local_stt_engine(model_size=model_size or Config.WHISPER_MODEL_SIZE)
        result = engine.transcribe_audio_buffer(
            audio_buffer=audio_buffer,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            min_silence_duration_ms=min_silence_duration_ms,
            temperature=temperature,
            best_of=best_of,
            beam_size=beam_size,
            patience=patience,
            length_penalty=length_penalty,
            compression_ratio_threshold=compression_ratio_threshold,
            no_speech_threshold=no_speech_threshold,
            cancellation_event=cancellation_event
        )
        
        if result:
            return result.get("text", "")
        return None
        
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            return None
        logging.error(f"{Fore.RED}Local STT buffer convenience function error: {e}{Fore.RESET}")
        return None 