import logging
import time
import threading
import queue
import os
import asyncio
from colorama import Fore, init
from voice_assistant.audio import record_audio, play_audio, record_audio_to_memory, play_audio_from_buffer, play_audio_from_buffer_interruptible
from voice_assistant.transcription import transcribe_audio, transcribe_audio_from_buffer
from voice_assistant.response_generation import generate_response
from voice_assistant.text_to_speech import text_to_speech, text_to_speech_buffer
from voice_assistant.utils import delete_file
from voice_assistant.config import Config
from voice_assistant.api_key_manager import get_transcription_api_key, get_response_api_key, get_tts_api_key
import torch
import numpy
import sounddevice as sd
from abc import ABC, abstractmethod
from collections import deque
import wave

# Import our new process manager and thread monitor
from process_manager import (
    process_manager, 
    start_llm_process, 
    start_tts_process, 
    cancel_all_ai_processes,
    ProcessType,
    InterruptibleLLMProcess,
    InterruptibleTTSProcess
)
from thread_monitor import (
    thread_monitor,
    start_thread_monitoring,
    stop_thread_monitoring,
    log_current_threads,
    log_thread_summary,
    log_cancellation_history,
    ThreadContext
)

# Helper function to handle both sync and async TTS models
async def generate_tts_async(model, api_key, text, output_file, local_model_path, cancellation_event):
    """Helper function to handle async TTS generation"""
    try:
        # For local models, we now use direct processing
        if model == "kokoro_local":
            # Local Kokoro is async
            result = await text_to_speech(model, api_key, text, output_file, local_model_path, cancellation_event)
        elif model == "kokoro_V2":
            # kokoro_V2 is async (HTTP-based, deprecated)
            result = await text_to_speech(model, api_key, text, output_file, local_model_path, cancellation_event)
        else:
            # Other models are sync
            result = text_to_speech(model, api_key, text, output_file, local_model_path, cancellation_event)
        return result
    except Exception as e:
        logging.error(f"TTS generation failed: {e}")
        return None

# Enhanced Observer Pattern Implementation with Process Management
class SpeechObserver(ABC):
    @abstractmethod
    def on_speech_detected(self):
        pass

class SpeechSubject:
    def __init__(self):
        self._observers = []
        self._speaking = False
        self.current_stt_cancellation = None  # Track current STT cancellation event
        self.interruption_requested = threading.Event()  # NEW: Signal to restart main loop
        
        # Anti-loop protection mechanisms
        self.last_speech_detection = 0  # Timestamp of last speech detection
        self.speech_detection_cooldown = 1.0  # Minimum 1 second between detections
        self.is_recording_phase = False  # Flag to indicate we're in recording mode
        self.recording_start_time = None  # When recording started
        
        # Emergency controls
        self.vad_enabled = True  # Master switch for VAD
        self.manual_override = False  # Manual override when user is speaking

    def attach(self, observer: SpeechObserver):
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: SpeechObserver):
        self._observers.remove(observer)

    def set_stt_cancellation(self, cancellation_event):
        """Set the current STT cancellation event"""
        self.current_stt_cancellation = cancellation_event

    def request_interruption(self):
        """Request interruption and restart of the main loop"""
        logging.info(f"{Fore.YELLOW}Interruption requested - signaling main loop to restart{Fore.RESET}")
        self.interruption_requested.set()

    def clear_interruption(self):
        """Clear the interruption signal"""
        self.interruption_requested.clear()

    def is_interruption_requested(self):
        """Check if interruption was requested"""
        return self.interruption_requested.is_set()
    
    def start_recording_phase(self):
        """Mark that we're entering recording phase (user should be speaking)"""
        self.is_recording_phase = True
        self.recording_start_time = time.time()
        self.manual_override = True  # Completely disable interruptions during recording
        logging.info(f"{Fore.CYAN}Recording phase started - VAD COMPLETELY DISABLED during recording{Fore.RESET}")
    
    def end_recording_phase(self):
        """Mark that we're exiting recording phase"""
        self.is_recording_phase = False
        self.recording_start_time = None
        self.manual_override = False  # Re-enable VAD after recording
        logging.info(f"{Fore.CYAN}Recording phase ended - VAD RE-ENABLED{Fore.RESET}")
    
    def disable_vad(self):
        """Emergency disable of VAD"""
        self.vad_enabled = False
        logging.info(f"{Fore.RED}🚫 VAD EMERGENCY DISABLED{Fore.RESET}")
    
    def enable_vad(self):
        """Re-enable VAD"""
        self.vad_enabled = True
        self.manual_override = False
        logging.info(f"{Fore.GREEN}✅ VAD RE-ENABLED{Fore.RESET}")
    
    def set_manual_override(self, override=True):
        """Set manual override for when user is definitely speaking"""
        self.manual_override = override
        status = "ENABLED" if override else "DISABLED"
        logging.info(f"{Fore.CYAN}🎤 Manual override {status}{Fore.RESET}")

    def notify_speech_detected(self):
        current_time = time.time()
        
        # Emergency override - completely disable VAD if requested
        if not self.vad_enabled or self.manual_override:
            logging.info(f"{Fore.RED}🚫 Speech detection DISABLED (vad_enabled={self.vad_enabled}, manual_override={self.manual_override}){Fore.RESET}")
            return
        
        # Anti-loop protection: Check cooldown period
        time_since_last = current_time - self.last_speech_detection
        if time_since_last < self.speech_detection_cooldown:
            logging.info(f"{Fore.YELLOW}🛡️ Speech detection BLOCKED - cooldown active ({time_since_last:.1f}s < {self.speech_detection_cooldown}s){Fore.RESET}")
            return
        
        # Don't interrupt if we're in recording phase (user is supposed to be speaking)
        if self.is_recording_phase:
            if self.recording_start_time and (current_time - self.recording_start_time) < 3.0:
                recording_duration = current_time - self.recording_start_time
                logging.info(f"{Fore.YELLOW}🎙️ Speech detection BLOCKED - recording phase active ({recording_duration:.1f}s){Fore.RESET}")
                return
        
        logging.info(f"{Fore.YELLOW}✅ Speech detected - notifying {len(self._observers)} observers (last: {time_since_last:.1f}s ago){Fore.RESET}")
        self.last_speech_detection = current_time
        
        # Cancel ongoing STT if active
        if self.current_stt_cancellation and not self.current_stt_cancellation.is_set():
            logging.info(f"{Fore.YELLOW}Cancelling ongoing STT transcription due to new speech{Fore.RESET}")
            self.current_stt_cancellation.set()
        
        # Notify observers (audio interruption)
        for observer in self._observers:
            observer.on_speech_detected()
        
        # Request interruption to restart the cycle
        self.request_interruption()
        
        logging.info(f"{Fore.GREEN}Speech interruption complete{Fore.RESET}")

    @property
    def speaking(self):
        return self._speaking

    @speaking.setter
    def speaking(self, value):
        self._speaking = value

class ProcessAwareAudioPlayer(SpeechObserver):
    """Enhanced AudioPlayer that integrates with process manager"""
    
    def __init__(self, speech_subject):
        self.speech_subject = speech_subject
        self.speech_subject.attach(self)
        self.should_stop = False
        self._current_stream = None
        self._audio_data = None
        self._stream_finished = threading.Event()
        self.current_frame = 0  # Track current position in audio
        self.current_process_id = None

    def on_speech_detected(self):
        if self.speech_subject.speaking:
            self.should_stop = True
            logging.info(f"{Fore.YELLOW}Speech detected during playback, interrupting audio...{Fore.RESET}")
            
            # Stop current audio stream
            if self._current_stream:
                try:
                    self._current_stream.abort()  # Use abort instead of stop for immediate effect
                    self._stream_finished.set()
                except Exception as e:
                    logging.error(f"Error stopping stream: {e}")
            
            # Log process cancellation info
            active_processes = process_manager.get_active_processes()
            if active_processes:
                logging.info(f"{Fore.CYAN}Active processes at interruption:{Fore.RESET}")
                for pid, info in active_processes.items():
                    logging.info(f"  - {info.process_type.value} (PID: {pid}, Thread: {info.thread_id})")

    def play(self, audio_file):
        """
        Play audio file using the most appropriate method.
        For local models, we use the standard audio player for simplicity.
        """
        self.should_stop = False
        self._stream_finished.clear()
        
        try:
            self.speech_subject.speaking = True
            logging.info(f"{Fore.GREEN}Starting audio playback: {audio_file}{Fore.RESET}")
            
            # For local processing, use the simple audio player
            play_audio(audio_file)
            
            logging.info(f"{Fore.GREEN}Audio playback completed successfully{Fore.RESET}")
            
        except Exception as e:
            logging.error(f"{Fore.RED}Error during audio playback: {e}{Fore.RESET}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.speech_subject.speaking = False
            self._stream_finished.set()
            logging.info(f"{Fore.CYAN}Audio playback cleanup complete{Fore.RESET}")


# Create global instances
speech_subject = SpeechSubject()
audio_player = ProcessAwareAudioPlayer(speech_subject)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress HTTP request logs from various libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)

# Initialize colorama
init(autoreset=True)

# Global variables and constants - Enhanced for longer sentences with loop prevention
SAMPLE_RATE = 16000  # 16000kHz sample rate
CHUNK_SIZE = 512  # Required chunk size for Silero VAD at 16kHz is 520
SPEECH_WINDOW = 8  # Balanced for stable detection without over-sensitivity
VAD_THRESHOLD = 0.6  # Increased to 0.6 to reduce false positives and loops
PRE_BUFFER_SIZE = int(1.5 * SAMPLE_RATE)  # Increased to 1.5 seconds of pre-buffer

is_playing = False
stop_event = threading.Event()
audio_q = queue.Queue()
vad_buffer = deque(maxlen=SPEECH_WINDOW)
pre_speech_buffer = deque(maxlen=PRE_BUFFER_SIZE)
calibrated = False

class AudioBuffer:
    def __init__(self):
        self.buffer = numpy.array([], dtype=numpy.float32)
        self.is_speech_started = False
        self.speech_buffer = []

    def add_samples(self, samples):
        self.buffer = numpy.append(self.buffer, samples)

    def get_chunks(self, chunk_size):
        chunks = []
        while len(self.buffer) >= chunk_size:
            chunk = self.buffer[:chunk_size]
            chunks.append(chunk)
            self.buffer = self.buffer[chunk_size:]
        return chunks

    def clear(self):
        self.buffer = numpy.array([], dtype=numpy.float32)
        self.speech_buffer = []
        self.is_speech_started = False

class VoiceDetector:
    def __init__(self):
        from silero_vad import load_silero_vad
        self.model = load_silero_vad()
        self.model.eval()
        self.audio_buffer = AudioBuffer()
        self.speech_detected = False
        self.speech_start_time = None
        self.reset_state()

    def reset_state(self):
        self.triggered = False
        self.voiced_frames = 0
        self.silent_frames = 0
        self.speech_detected = False
        self.speech_start_time = None
        self.audio_buffer.clear()

    def is_speech(self, audio_chunk):
        try:
            if len(audio_chunk) != CHUNK_SIZE:
                return False

            audio_tensor = torch.from_numpy(audio_chunk).float()
            audio_tensor = audio_tensor.unsqueeze(0)
 
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, SAMPLE_RATE)
                return speech_prob.item() > VAD_THRESHOLD

        except Exception as e:
            logging.error(f"Error in speech detection: {e}")
            return False

    def save_audio(self, filename='test.wav'):
        if len(self.audio_buffer.speech_buffer) > 0:
            try:
                audio_data = numpy.concatenate(self.audio_buffer.speech_buffer)
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(SAMPLE_RATE)
                    # Convert float32 to int16
                    audio_data_int = (audio_data * 32767).astype(numpy.int16)
                    wf.writeframes(audio_data_int.tobytes())
                return True
            except Exception as e:
                logging.error(f"Error saving audio: {e}")
                return False
        return False

    def get_audio_buffer(self):
        """
        Get audio data as a BytesIO buffer in MP3 format instead of saving to file.
        Returns the audio buffer directly for STT processing.
        """
        if len(self.audio_buffer.speech_buffer) > 0:
            try:
                import pydub
                from io import BytesIO
                
                # Concatenate all speech buffer data
                audio_data = numpy.concatenate(self.audio_buffer.speech_buffer)
                
                # Convert float32 to int16
                audio_data_int = (audio_data * 32767).astype(numpy.int16)
                
                # Create pydub AudioSegment from raw audio data
                audio_segment = pydub.AudioSegment(
                    audio_data_int.tobytes(),
                    frame_rate=SAMPLE_RATE,
                    sample_width=2,  # 16-bit = 2 bytes
                    channels=1
                )
                
                # Export to MP3 format in memory buffer
                mp3_buffer = BytesIO()
                audio_segment.export(mp3_buffer, format="mp3", bitrate="128k", parameters=["-ar", "22050", "-ac", "1"])
                mp3_buffer.seek(0)  # Reset buffer position to beginning
                
                logging.info(f"{Fore.GREEN}Audio buffer created successfully: {len(audio_data)} samples -> MP3 buffer{Fore.RESET}")
                return mp3_buffer
                
            except Exception as e:
                logging.error(f"{Fore.RED}Error creating audio buffer: {e}{Fore.RESET}")
                return None
        return None

    def process_audio(self, audio_chunk):
        # Always add incoming audio to pre-speech buffer
        for sample in audio_chunk:
            pre_speech_buffer.append(sample)

        # Add new audio to buffer
        self.audio_buffer.add_samples(audio_chunk)
        chunks = self.audio_buffer.get_chunks(CHUNK_SIZE)
        
        for chunk in chunks:
            is_speech = self.is_speech(chunk)
            vad_buffer.append(is_speech)
            
            # Require at least 3 consecutive speech frames to reduce false positives
            consecutive_speech_frames = sum(1 for val in list(vad_buffer)[-3:] if val)
            
            if consecutive_speech_frames >= 3:  # More stringent requirement
                if not self.speech_detected:
                    # Speech just started - include pre-buffer
                    self.speech_detected = True
                    self.speech_start_time = time.time()
                    self.audio_buffer.is_speech_started = True
                    # Add pre-buffer to speech buffer
                    self.audio_buffer.speech_buffer.append(numpy.array(pre_speech_buffer))
                
                # Continue adding audio while speech is detected
                self.audio_buffer.speech_buffer.append(chunk)
                return True
            
            elif self.speech_detected:
                # Add a bit more audio after speech ends
                self.audio_buffer.speech_buffer.append(chunk)
                # Check if we've captured enough silence - optimized for faster response
                if len(self.audio_buffer.speech_buffer) > 15:  # About 300ms of silence - much faster response
                    # Audio recording is complete - reset state
                    # Note: The get_audio_buffer() method can be called separately when needed
                    self.reset_state()
                    return False  # Speech has ended
        
        return self.speech_detected

voice_detector = VoiceDetector()

# Function to check if the assistant is speaking
def is_assistant_speaking():
    return speech_subject.speaking

def vad_callback(indata, frames, time_info, status):
    try:
        if status:
            logging.warning(f"VAD callback status: {status}")
        
        # Convert to the expected format and add to queue
        audio_data = indata[:, 0] if indata.ndim > 1 else indata
        audio_q.put(audio_data.copy())

    except Exception as e:
        logging.error(f"Error in VAD callback: {e}")

def silero_vad_listener():
    """
    Enhanced Silero VAD listener with process management integration
    """
    try:
        # Initialize input stream with smaller blocksize
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',      
            blocksize=CHUNK_SIZE,  # Use the same size as required by Silero VAD
            callback=vad_callback
        ) as stream:
            logging.info("VAD listener started with process management")
            
            while not stop_event.is_set():
                try:
                    # Get audio frame with a short timeout
                    frame = audio_q.get(timeout=0.1)
                    
                    # Process frame for speech detection
                    if voice_detector.process_audio(frame):
                        logging.info(Fore.GREEN + "Speech detected!" + Fore.RESET)
                        
                        # Log current active processes before interruption
                        active_processes = process_manager.get_active_processes()
                        if active_processes:
                            logging.info(f"{Fore.CYAN}Interrupting {len(active_processes)} active processes{Fore.RESET}")
                        
                        # If assistant is speaking or processes are running, interrupt them
                        if is_playing or active_processes or speech_subject.speaking:
                            logging.info(Fore.YELLOW + "Interrupting current operations..." + Fore.RESET)
                            speech_subject.notify_speech_detected()
                            # Don't set stop_event globally - let the main loop handle restarts
                            time.sleep(0.1)  # Small delay to allow cleanup
                            voice_detector.reset_state()
                            
                            # Log process status after interruption
                            process_manager.log_process_status()
                            
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Error processing audio frame: {e}")
                    time.sleep(0.1)
                    
    except Exception as e:
        logging.error(f"Error in VAD listener: {e}")
    finally:
        logging.info("VAD listener stopped")

# Function to stop audio playback when new speech is detected
def stop_playing_audio():
    """
    Function to gracefully stop audio playback
    """
    global is_playing
    is_playing = False
    speech_subject.speaking = False

def play_audio_in_background(output_file_updated):
    global is_playing
    thread_id = threading.current_thread().ident
    thread_name = threading.current_thread().name
    
    try:
        is_playing = True
        logging.info(f"{Fore.CYAN}Audio playback thread started (ID: {thread_id}, Name: {thread_name}){Fore.RESET}")
        
        # Check for interruption before starting playback
        if speech_subject.is_interruption_requested():
            logging.info(f"{Fore.YELLOW}Audio playback thread cancelled before starting due to interruption{Fore.RESET}")
            return
        
        audio_player.play(output_file_updated)
        logging.info(f"{Fore.GREEN}Audio playback thread completed successfully{Fore.RESET}")
    except Exception as e:
        logging.error(f"{Fore.RED}Error in audio playback thread: {e}{Fore.RESET}")
        import traceback
        logging.error(f"Thread traceback: {traceback.format_exc()}")
    finally:
        is_playing = False
        logging.info(f"{Fore.CYAN}Audio playback thread finished (ID: {thread_id}){Fore.RESET}")

# Wrapper functions for process-managed AI operations with thread monitoring
def process_managed_llm_response(model, api_key, chat_history, local_model_path=None):
    """Generate LLM response using process manager with thread monitoring"""
    with ThreadContext("LLM_Generation", "llm_inference"):
        # Create a shared cancellation event
        cancellation_event = threading.Event()
        
        def llm_wrapper():
            with ThreadContext("LLM_Execution", "llm_inference"):
                return generate_response(model, api_key, chat_history, local_model_path, 
                                       cancellation_event=cancellation_event)
        
        # Create process with the cancellation event
        process_id = process_manager.generate_process_id(ProcessType.LLM_INFERENCE)
        process = InterruptibleLLMProcess(process_id, llm_wrapper)
        process.is_cancelled = cancellation_event  # Link the events
        
        # Start the process
        process_manager.start_process(process)
        
        # Wait for completion with timeout
        if process.wait_for_completion(timeout=30):
            result = process.result
            if result:
                logging.info(f"{Fore.GREEN}LLM process completed successfully{Fore.RESET}")
            return result
        else:
            logging.warning(f"{Fore.YELLOW}LLM process timed out{Fore.RESET}")
            process_manager.cancel_process(process_id)
            thread_monitor.log_thread_cancellation(
                threading.current_thread().ident,
                threading.current_thread().name,
                "LLM timeout"
            )
            return None

def process_managed_tts_generation(model, api_key, text, output_file, local_model_path=None):
    """Generate TTS using process manager with thread monitoring"""
    with ThreadContext("TTS_Generation", "tts_generation"):
        logging.info(f"{Fore.CYAN}Creating TTS cancellation event...{Fore.RESET}")
        # Create a shared cancellation event
        cancellation_event = threading.Event()
        
        def tts_wrapper():
            logging.info(f"{Fore.CYAN}TTS wrapper function started{Fore.RESET}")
            with ThreadContext("TTS_Execution", "tts_generation"):
                return text_to_speech(model, api_key, text, output_file, local_model_path,
                                    cancellation_event=cancellation_event)
        
        logging.info(f"{Fore.CYAN}Creating TTS process...{Fore.RESET}")
        # Create process with the cancellation event
        process_id = process_manager.generate_process_id(ProcessType.TTS_GENERATION)
        process = InterruptibleTTSProcess(process_id, tts_wrapper)
        process.is_cancelled = cancellation_event  # Link the events
        
        logging.info(f"{Fore.CYAN}Starting TTS process {process_id}...{Fore.RESET}")
        # Start the process
        process_manager.start_process(process)
        
        logging.info(f"{Fore.CYAN}Waiting for TTS process completion...{Fore.RESET}")
        # Wait for completion with timeout
        if process.wait_for_completion(timeout=30):
            result = process.result
            if result:
                logging.info(f"{Fore.GREEN}TTS process completed successfully{Fore.RESET}")
            return result
        else:
            logging.warning(f"{Fore.YELLOW}TTS process timed out{Fore.RESET}")
            process_manager.cancel_process(process_id)
            thread_monitor.log_thread_cancellation(
                threading.current_thread().ident,
                threading.current_thread().name,
                "TTS timeout"
            )
            return None

# Main function for speech-to-text, response generation, and text-to-speech
async def listen_and_process_speech():
    chat_history = [{"role": "system", "content": """ You are a helpful Assistant called Olivia. 
         You are friendly and fun and you will help the users with their requests.
         Your answers are should be concise and short, maximum 3 lines.Please provide a clean response with no special characters like '*', '#', etc.
Just give me the plain text"""}]
    
    global calibrated
    calibration_count = 0
    max_calibration_attempts = 3
    
    # Display configuration info
    config_info = Config.get_local_config_info()
    logging.info(f"{Fore.GREEN}=== LOCAL PROCESSING CONFIGURATION ==={Fore.RESET}")
    logging.info(f"{Fore.CYAN}STT Model: {config_info['transcription']['model']} (Local: {config_info['transcription']['local_processing']}){Fore.RESET}")
    logging.info(f"{Fore.CYAN}TTS Model: {config_info['tts']['model']} (Local: {config_info['tts']['local_processing']}){Fore.RESET}")
    logging.info(f"{Fore.CYAN}LLM Model: {config_info['response']['model']}{Fore.RESET}")
    logging.info(f"{Fore.GREEN}=== ALL HTTP DEPENDENCIES REMOVED ==={Fore.RESET}")
    
    # Start VAD listener in a separate thread
    vad_thread = threading.Thread(target=silero_vad_listener, daemon=True)
    vad_thread.start()
    
    while True:
        try:
            # Clear any previous interruption signals at the start of each cycle
            speech_subject.clear_interruption()
            
            # Add a small delay at start of each cycle to prevent immediate re-triggering
            time.sleep(0.5)
            
            # Calibrate for ambient noise only at the start or after errors
            if not calibrated and calibration_count < max_calibration_attempts:
                logging.info("Calibrating for ambient noise...")
                time.sleep(1)  # Give time for system to settle
                calibrated = True
                calibration_count += 1

            # Start recording phase - disable aggressive speech detection
            speech_subject.start_recording_phase()
            
            try:
                # Record audio from the microphone with optimized timing for fast response
                start_record_time = time.time()
                # Smart recording parameters - optimized for fast response when user stops speaking
                logging.info(f"{Fore.CYAN}Starting audio recording...{Fore.RESET}")
                audio_buffer = record_audio_to_memory(
                    timeout=30,              # Increased wait time for speech to start
                    phrase_time_limit=30,    # Increased max recording time (30 seconds for longer conversations)
                    energy_threshold=300,    # Lower threshold for better sensitivity (was 500)
                    pause_threshold=2.0,     # Allow 2 seconds for breathing - good balance
                    retries=3,
                    dynamic_energy_threshold=True,
                    calibration_duration=1.0) # Slightly longer calibration for better ambient noise detection
                record_duration = time.time() - start_record_time
                
                if not audio_buffer:
                    logging.warning(f"{Fore.YELLOW}Failed to record audio, retrying...{Fore.RESET}")
                    continue
                
                # Intelligent recording feedback
                if record_duration < 5:
                    logging.info(f"{Fore.GREEN}⚡ Quick recording: {record_duration:.2f}s (user stopped speaking early){Fore.RESET}")
                elif record_duration >= 18:
                    logging.info(f"{Fore.BLUE}🗣️ Long recording: {record_duration:.2f}s (full sentence captured){Fore.RESET}")
                else:
                    logging.info(f"{Fore.CYAN}📝 Recording completed: {record_duration:.2f}s{Fore.RESET}")
            finally:
                # Always end recording phase
                speech_subject.end_recording_phase()

            # Check for interruption after recording
            if speech_subject.is_interruption_requested():
                logging.info(f"{Fore.YELLOW}Interruption detected after recording, restarting cycle...{Fore.RESET}")
                continue

            # Get the API key for transcription
            transcription_api_key = get_transcription_api_key()
            
            # Transcribe the audio buffer with interruption support
            logging.info(f"{Fore.CYAN}Starting STT transcription using {Config.TRANSCRIPTION_MODEL}...{Fore.RESET}")
            stt_cancellation = threading.Event()
            stt_result = {'transcription': None}
            
            # Register this STT operation for potential cancellation
            speech_subject.set_stt_cancellation(stt_cancellation)
            
            def stt_worker():
                try:
                    result = transcribe_audio_from_buffer(Config.TRANSCRIPTION_MODEL, transcription_api_key, audio_buffer, Config.LOCAL_MODEL_PATH, stt_cancellation)
                    stt_result['transcription'] = result
                except Exception as e:
                    logging.error(f"{Fore.RED}STT worker error: {e}{Fore.RESET}")
                    stt_result['transcription'] = None
            
            # Run STT in a thread to allow for potential interruption
            stt_thread = threading.Thread(target=stt_worker, name="STT_Worker")
            stt_thread.start()
            logging.info(f"{Fore.CYAN}STT thread started (ID: {stt_thread.ident}){Fore.RESET}")
            
            # Wait for STT completion with timeout
            stt_thread.join(timeout=30)  # 30 second timeout for STT
            
            # Clear the STT cancellation reference
            speech_subject.set_stt_cancellation(None)
            
            if stt_thread.is_alive():
                logging.warning(f"{Fore.YELLOW}STT transcription timed out, cancelling...{Fore.RESET}")
                stt_cancellation.set()
                stt_thread.join(timeout=2)  # Give it 2 seconds to cleanup
                user_input = None
            else:
                user_input = stt_result['transcription']
                if user_input:
                    logging.info(f"{Fore.GREEN}STT transcription completed successfully{Fore.RESET}")
                elif stt_cancellation.is_set():
                    logging.info(f"{Fore.YELLOW}STT transcription was cancelled{Fore.RESET}")
                else:
                    logging.info(f"{Fore.YELLOW}STT transcription returned no result{Fore.RESET}")

            # Check for interruption after STT
            if speech_subject.is_interruption_requested():
                logging.info(f"{Fore.YELLOW}Interruption detected after STT, restarting cycle...{Fore.RESET}")
                continue

            # Check if the transcription is empty
            if not user_input:
                logging.info("No transcription was returned. Starting recording again.")
                continue
                
            logging.info(Fore.GREEN + "You said: " + user_input + Fore.RESET)

            # Check for exit keywords
            if "goodbye" in user_input.lower() or "arrivederci" in user_input.lower():
                break
            
            # Check for VAD control commands
            if "disable vad" in user_input.lower() or "stop interrupting" in user_input.lower():
                speech_subject.disable_vad()
                logging.info(f"{Fore.GREEN}VAD disabled by voice command{Fore.RESET}")
                continue
            elif "enable vad" in user_input.lower() or "start interrupting" in user_input.lower():
                speech_subject.enable_vad()
                logging.info(f"{Fore.GREEN}VAD enabled by voice command{Fore.RESET}")
                continue

            # Append the user's input to the chat history
            chat_history.append({"role": "user", "content": user_input})

            # Get the API key for response generation
            response_api_key = get_response_api_key()

            # Check for interruption before LLM generation
            if speech_subject.is_interruption_requested():
                logging.info(f"{Fore.YELLOW}Interruption detected before LLM generation, restarting cycle...{Fore.RESET}")
                continue

            # Generate a response using direct method (bypassing process manager for now)
            logging.info(f"{Fore.CYAN}Generating response using {Config.RESPONSE_MODEL}...{Fore.RESET}")
            try:
                response_text = generate_response(Config.RESPONSE_MODEL, response_api_key, chat_history, Config.LOCAL_MODEL_PATH)
                logging.info(f"{Fore.GREEN}LLM generation completed{Fore.RESET}")
            except Exception as e:
                logging.error(f"{Fore.RED}LLM generation failed: {e}{Fore.RESET}")
                response_text = None
            
            # Check for interruption after LLM generation
            if speech_subject.is_interruption_requested():
                logging.info(f"{Fore.YELLOW}Interruption detected after LLM generation, restarting cycle...{Fore.RESET}")
                continue
            
            if response_text is None:
                logging.info(f"{Fore.YELLOW}LLM response was cancelled or failed{Fore.RESET}")
                continue
                
            logging.info(Fore.CYAN + "Response: " + response_text + Fore.RESET)

            # Append the assistant's response to the chat history
            if len(chat_history) >= 5:
                del(chat_history[1])  # Limit chat history size
            chat_history.append({"role": "assistant", "content": response_text})

            # Check for interruption before TTS generation
            if speech_subject.is_interruption_requested():
                logging.info(f"{Fore.YELLOW}Interruption detected before TTS generation, restarting cycle...{Fore.RESET}")
                continue

            # Get the API key for TTS
            tts_api_key = get_tts_api_key()

            # Generate TTS using in-memory processing
            logging.info(f"{Fore.CYAN}Generating TTS using {Config.TTS_MODEL}...{Fore.RESET}")
            
            try:
                # Generate TTS audio buffer directly
                tts_audio_buffer = await text_to_speech_buffer(Config.TTS_MODEL, tts_api_key, response_text, Config.LOCAL_MODEL_PATH, None)
                
                # Check for interruption after TTS generation
                if speech_subject.is_interruption_requested():
                    logging.info(f"{Fore.YELLOW}Interruption detected after TTS generation, restarting cycle...{Fore.RESET}")
                    continue
                
                if tts_audio_buffer:
                    logging.info(f"{Fore.GREEN}TTS generation completed successfully{Fore.RESET}")
                else:
                    logging.warning(f"{Fore.YELLOW}TTS generation returned no result{Fore.RESET}")
                    continue
                    
            except Exception as e:
                logging.error(f"{Fore.RED}TTS generation failed: {e}{Fore.RESET}")
                continue

            # Play the generated speech audio directly from buffer in a separate thread
            def play_audio_in_background_buffer(audio_buffer):
                try:
                    global is_playing
                    is_playing = True
                    speech_subject.speaking = True
                    
                    # Use the interruption event from speech_subject to allow interruption
                    completed = play_audio_from_buffer_interruptible(audio_buffer, speech_subject.interruption_requested)
                    
                    if not completed:
                        logging.info(f"{Fore.YELLOW}Audio playback was interrupted{Fore.RESET}")
                    
                except Exception as e:
                    logging.error(f"{Fore.RED}Error playing audio: {e}{Fore.RESET}")
                finally:
                    is_playing = False
                    speech_subject.speaking = False
            
            # Start a new thread for audio playback
            playback_thread = threading.Thread(target=play_audio_in_background_buffer, args=(tts_audio_buffer,), name="AudioPlayback")
            playback_thread.start()
            logging.info(f"{Fore.CYAN}Started audio playback thread: {playback_thread.name} (ID: {playback_thread.ident}){Fore.RESET}")

            # Wait for playback to complete or be interrupted
            playback_thread.join(timeout=30)  # Add timeout to prevent hanging
            
            # Check for interruption after playback
            if speech_subject.is_interruption_requested():
                logging.info(f"{Fore.YELLOW}Interruption detected after audio playback, restarting cycle...{Fore.RESET}")
                # No files to clean up since we're using in-memory processing
                continue
            
            if playback_thread.is_alive():
                logging.warning(f"{Fore.YELLOW}Audio playback thread timed out after 30 seconds and is still alive{Fore.RESET}")
            else:
                logging.info(f"{Fore.GREEN}Audio playback thread completed normally{Fore.RESET}")

            # No file cleanup needed since we're using in-memory processing
            logging.info(f"{Fore.GREEN}Voice assistant cycle completed successfully - no files to clean up{Fore.RESET}")

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            calibrated = False
            calibration_count = 0
            time.sleep(1)

# Enhanced main function with process management and thread monitoring
if __name__ == "__main__":
    try:
        logging.info(f"{Fore.GREEN}Starting Voice Assistant with Local Processing (No HTTP Dependencies){Fore.RESET}")
        logging.info(f"{Fore.CYAN}Process manager initialized{Fore.RESET}")
        
        # Start thread monitoring
        start_thread_monitoring(interval=10.0)  # Monitor every 10 seconds
        logging.info(f"{Fore.CYAN}Thread monitoring started{Fore.RESET}")
        
        # Log initial thread state
        log_thread_summary()
        
        # Run the async main function
        asyncio.run(listen_and_process_speech())
        
    except KeyboardInterrupt:
        logging.info(f"{Fore.YELLOW}Shutting down...{Fore.RESET}")
        stop_event.set()
        cancel_all_ai_processes()
        
        # Log final states
        process_manager.log_process_status()
        log_thread_summary()
        log_cancellation_history()
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        
    finally:
        stop_event.set()
        cancel_all_ai_processes()
        stop_thread_monitoring()
        logging.info(f"{Fore.GREEN}Voice Assistant stopped{Fore.RESET}")

