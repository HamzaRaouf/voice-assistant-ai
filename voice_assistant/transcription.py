# voice_assistant/transcription.py

import json
import logging
import requests
import time
import threading
import tempfile
import os
from io import BytesIO

from colorama import Fore, init
from openai import OpenAI
from groq import Groq
from deepgram import DeepgramClient,PrerecordedOptions,FileSource

# Suppress HTTP request logs from various libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)

# Import the new local STT engine
from .local_stt import get_local_stt_engine, transcribe_audio_local, transcribe_audio_buffer_local
from .config import Config

fast_url = "http://localhost:8000"  #  previous 
# fast_url = "http://192.168.100.53:8000"  #  previous 
# fast_url = "https://2e5ec5937ad7.ngrok-free.app"
# checked_fastwhisperapi = False  previous
checked_fastwhisperapi = True

def check_fastwhisperapi():
    """Check if the FastWhisper API is running."""
    global checked_fastwhisperapi, fast_url
    if not checked_fastwhisperapi:
        infopoint = f"{fast_url}/info"
        try:
            response = requests.get(infopoint)
            if response.status_code != 200:
                raise Exception("FastWhisperAPI is not running")
        except Exception:
            raise Exception("FastWhisperAPI is not running")
        checked_fastwhisperapi = True

def transcribe_audio(model, api_key, audio_file_path, local_model_path=None, cancellation_event=None):
    """
    Transcribe an audio file using the specified model with cancellation support.
    
    Args:
        model (str): The model to use for transcription ('openai', 'groq', 'deepgram', 'fastwhisperapi', 'fastwhisper_local', 'local').
        api_key (str): The API key for the transcription service.
        audio_file_path (str): The path to the audio file to transcribe.
        local_model_path (str): The path to the local model (if applicable).
        cancellation_event (threading.Event): Event to check for cancellation

    Returns:
        str: The transcribed text, or None if cancelled.
    """
    try:
        # Check for cancellation before starting
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}STT transcription cancelled before starting{Fore.RESET}")
            return None
            
        if model == 'openai':
            return _transcribe_with_openai(api_key, audio_file_path, cancellation_event)
        elif model == 'groq':
            return _transcribe_with_groq(api_key, audio_file_path, cancellation_event)
        elif model == 'deepgram':
            return _transcribe_with_deepgram(api_key, audio_file_path, cancellation_event)
        elif model == 'fastwhisperapi':
            return _transcribe_with_fastwhisperapi(audio_file_path, cancellation_event)
        elif model == 'fastwhisper_local':
            # NEW: Use local faster-whisper directly without HTTP
            return _transcribe_with_fastwhisper_local(audio_file_path, cancellation_event)
        elif model == 'local':
            # Placeholder for local STT model transcription
            if cancellation_event and cancellation_event.is_set():
                return None
            return "Transcribed text from local model"
        else:
            raise ValueError("Unsupported transcription model")
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}STT transcription cancelled due to interruption{Fore.RESET}")
            return None
        logging.error(f"{Fore.RED}Failed to transcribe audio: {e}{Fore.RESET}")
        raise Exception("Error in transcribing audio")

def transcribe_audio_from_buffer(model, api_key, audio_buffer, local_model_path=None, cancellation_event=None):
    """
    Transcribe audio data from a BytesIO buffer using the specified model with cancellation support.
    
    Args:
        model (str): The model to use for transcription ('openai', 'groq', 'deepgram', 'fastwhisperapi', 'fastwhisper_local', 'local').
        api_key (str): The API key for the transcription service.
        audio_buffer (BytesIO): The audio data buffer to transcribe.
        local_model_path (str): The path to the local model (if applicable).
        cancellation_event (threading.Event): Event to check for cancellation

    Returns:
        str: The transcribed text, or None if cancelled.
    """
    try:
        # Check for cancellation before starting
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}STT transcription cancelled before starting{Fore.RESET}")
            return None
            
        if model == 'openai':
            return _transcribe_with_openai_buffer(api_key, audio_buffer, cancellation_event)
        elif model == 'groq':
            return _transcribe_with_groq_buffer(api_key, audio_buffer, cancellation_event)
        elif model == 'deepgram':
            return _transcribe_with_deepgram_buffer(api_key, audio_buffer, cancellation_event)
        elif model == 'fastwhisperapi':
            return _transcribe_with_fastwhisperapi_buffer(audio_buffer, cancellation_event)
        elif model == 'fastwhisper_local':
            # NEW: Use local faster-whisper directly without HTTP
            return _transcribe_with_fastwhisper_local_buffer(audio_buffer, cancellation_event)
        elif model == 'local':
            # Placeholder for local STT model transcription
            if cancellation_event and cancellation_event.is_set():
                return None
            return "Transcribed text from local model"
        else:
            raise ValueError("Unsupported transcription model")
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}STT transcription cancelled due to interruption{Fore.RESET}")
            return None
        logging.error(f"{Fore.RED}Failed to transcribe audio: {e}{Fore.RESET}")
        raise Exception("Error in transcribing audio")

def _transcribe_with_fastwhisper_local(audio_file_path, cancellation_event=None):
    """
    NEW: Local faster-whisper transcription without HTTP calls.
    This uses the local STT engine directly for better performance and no network dependency.
    """
    try:
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Local FastWhisper STT cancelled before starting{Fore.RESET}")
            return None
        
        logging.info(f"{Fore.CYAN}Using local faster-whisper for transcription: {audio_file_path}{Fore.RESET}")
        
        # Use the local STT engine with appropriate parameters
        result = transcribe_audio_local(
            audio_file_path=audio_file_path,
            model_size=Config.WHISPER_MODEL_SIZE,  # Use configured model size instead of hardcoded "base"
            language="en",
            initial_prompt=None,
            word_timestamps=False,
            vad_filter=True,
            min_silence_duration_ms=1000,
            cancellation_event=cancellation_event
        )
        
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Local FastWhisper STT cancelled after processing{Fore.RESET}")
            return None
        
        if result:
            logging.info(f"{Fore.GREEN}Local FastWhisper transcription completed successfully{Fore.RESET}")
            return result
        else:
            logging.warning(f"{Fore.YELLOW}Local FastWhisper transcription returned no result{Fore.RESET}")
            return None
            
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Local FastWhisper STT cancelled due to interruption{Fore.RESET}")
            return None
        logging.error(f"{Fore.RED}Local FastWhisper transcription error: {e}{Fore.RESET}")
        raise

def _transcribe_with_openai(api_key, audio_file_path, cancellation_event=None):
    """OpenAI transcription with cancellation support"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    client = OpenAI(api_key=api_key)
    with open(audio_file_path, "rb") as audio_file:
        # Check cancellation before API call
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}OpenAI STT cancelled before API call{Fore.RESET}")
            return None
            
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language='en'
        )
        
        # Check cancellation after API call
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}OpenAI STT cancelled after API call{Fore.RESET}")
            return None
            
    return transcription.text


def _transcribe_with_groq(api_key, audio_file_path, cancellation_event=None):
    """Groq transcription with cancellation support"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    client = Groq(api_key=api_key)
    with open(audio_file_path, "rb") as audio_file:
        # Check cancellation before API call
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Groq STT cancelled before API call{Fore.RESET}")
            return None
            
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            language='en'
        )
        
        # Check cancellation after API call
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Groq STT cancelled after API call{Fore.RESET}")
            return None
            
    return transcription.text

def _transcribe_with_deepgram(api_key, audio_file_path, cancellation_event=None):
    """Deepgram transcription with cancellation support"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    try:
        deepgram = DeepgramClient(api_key)

        with open(audio_file_path, "rb") as file:
            buffer_data = file.read()

        # Check cancellation before API call
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Deepgram STT cancelled before API call{Fore.RESET}")
            return None

        payload = {"buffer": buffer_data}
        options = PrerecordedOptions(model="nova-2", smart_format=True)
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        
        # Check cancellation after API call
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Deepgram STT cancelled after API call{Fore.RESET}")
            return None
            
        data = json.loads(response.to_json())
        transcript = data['results']['channels'][0]['alternatives'][0]['transcript']
        return transcript
        
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Deepgram STT cancelled due to interruption{Fore.RESET}")
            return None
        logging.error(f"{Fore.RED}Deepgram transcription error: {e}{Fore.RESET}")
        raise


def _transcribe_with_fastwhisperapi(audio_file_path, cancellation_event=None):
    """FastWhisperAPI transcription with cancellation support (HTTP-based - deprecated)"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    try:
        check_fastwhisperapi()
        endpoint = f"{fast_url}/v1/transcriptions"

        # Check cancellation before preparing request
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}FastWhisperAPI STT cancelled before request{Fore.RESET}")
            return None

        files = {'file': (audio_file_path, open(audio_file_path, 'rb'))}
        data = {
            'model': "base",
            'language': "en",
            'initial_prompt': None,
            'vad_filter': True,
        }
        headers = {'Authorization': 'Bearer dummy_api_key'}

        # Make request with timeout to allow cancellation
        response = requests.post(endpoint, files=files, data=data, headers=headers, timeout=30)
        
        # Check cancellation after API call
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}FastWhisperAPI STT cancelled after API call{Fore.RESET}")
            return None
            
        response_json = response.json()
        return response_json.get('text', 'No text found in the response.')
        
    except requests.exceptions.RequestException as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}FastWhisperAPI STT cancelled due to interruption{Fore.RESET}")
            return None
        raise e
    finally:
        # Make sure to close the file
        try:
            files['file'][1].close()
        except:
            pass

def _transcribe_with_openai_buffer(api_key, audio_buffer, cancellation_event=None):
    """OpenAI transcription with cancellation support from audio buffer"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    client = OpenAI(api_key=api_key)
    
    # Reset buffer position to beginning
    audio_buffer.seek(0)
    
    # Check cancellation before API call
    if cancellation_event and cancellation_event.is_set():
        logging.info(f"{Fore.YELLOW}OpenAI STT cancelled before API call{Fore.RESET}")
        return None
        
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_buffer,
        language='en'
    )
    
    # Check cancellation after API call
    if cancellation_event and cancellation_event.is_set():
        logging.info(f"{Fore.YELLOW}OpenAI STT cancelled after API call{Fore.RESET}")
        return None
        
    return transcription.text

def _transcribe_with_groq_buffer(api_key, audio_buffer, cancellation_event=None):
    """Groq transcription with cancellation support from audio buffer"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    client = Groq(api_key=api_key)
    
    # Reset buffer position to beginning
    audio_buffer.seek(0)
    
    # Check cancellation before API call
    if cancellation_event and cancellation_event.is_set():
        logging.info(f"{Fore.YELLOW}Groq STT cancelled before API call{Fore.RESET}")
        return None
        
    transcription = client.audio.transcriptions.create(
        file=audio_buffer,
        model="whisper-large-v3",
        language='en'
    )
    
    # Check cancellation after API call
    if cancellation_event and cancellation_event.is_set():
        logging.info(f"{Fore.YELLOW}Groq STT cancelled after API call{Fore.RESET}")
        return None
        
    return transcription.text

def _transcribe_with_deepgram_buffer(api_key, audio_buffer, cancellation_event=None):
    """Deepgram transcription with cancellation support from audio buffer"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    try:
        deepgram = DeepgramClient(api_key)

        # Reset buffer position to beginning and read data
        audio_buffer.seek(0)
        buffer_data = audio_buffer.read()

        # Check cancellation before API call
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Deepgram STT cancelled before API call{Fore.RESET}")
            return None

        payload = {"buffer": buffer_data}
        options = PrerecordedOptions(model="nova-2", smart_format=True)
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        
        # Check cancellation after API call
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Deepgram STT cancelled after API call{Fore.RESET}")
            return None
            
        data = json.loads(response.to_json())
        transcript = data['results']['channels'][0]['alternatives'][0]['transcript']
        return transcript
        
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Deepgram STT cancelled due to interruption{Fore.RESET}")
            return None
        logging.error(f"{Fore.RED}Deepgram transcription error: {e}{Fore.RESET}")
        raise

def _transcribe_with_fastwhisperapi_buffer(audio_buffer, cancellation_event=None):
    """FastWhisperAPI transcription with cancellation support from audio buffer (HTTP-based - deprecated)"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    try:
        check_fastwhisperapi()
        endpoint = f"{fast_url}/v1/transcriptions"

        # Check cancellation before preparing request
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}FastWhisperAPI STT cancelled before request{Fore.RESET}")
            return None

        # Reset buffer position to beginning
        audio_buffer.seek(0)
        files = {'file': ('audio.mp3', audio_buffer, 'audio/mpeg')}
        data = {
            'model': "base",
            'language': "en",
            'initial_prompt': None,
            'vad_filter': True,
        }
        headers = {'Authorization': 'Bearer dummy_api_key'}

        # Make request with timeout to allow cancellation
        response = requests.post(endpoint, files=files, data=data, headers=headers, timeout=30)
        
        # Check cancellation after API call
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}FastWhisperAPI STT cancelled after API call{Fore.RESET}")
            return None
            
        response_json = response.json()
        return response_json.get('text', 'No text found in the response.')
        
    except requests.exceptions.RequestException as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}FastWhisperAPI STT cancelled due to interruption{Fore.RESET}")
            return None
        raise e

def _transcribe_with_fastwhisper_local_buffer(audio_buffer, cancellation_event=None):
    """
    NEW: Local faster-whisper transcription from audio buffer without HTTP calls.
    This uses the local STT engine directly for better performance and no network dependency.
    Now processes audio directly without saving to temporary files.
    """
    try:
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Local FastWhisper STT cancelled before starting{Fore.RESET}")
            return None
        
        logging.info(f"{Fore.CYAN}Using local faster-whisper for direct buffer transcription{Fore.RESET}")
        
        # Use the new direct buffer transcription with enhanced parameters
        result = transcribe_audio_buffer_local(
            audio_buffer=audio_buffer,
            model_size=Config.WHISPER_MODEL_SIZE,  # Use configured model size instead of hardcoded "base"
            language="en",
            initial_prompt=None,
            word_timestamps=False,
            vad_filter=True,
            min_silence_duration_ms=500,  # Reduced from 1000ms - less aggressive VAD
            temperature=0.0,  # Deterministic output
            best_of=5,  # Generate 5 candidates and pick the best
            beam_size=5,  # Beam search with size 5
            patience=1.0,  # Patience for beam search
            length_penalty=1.0,  # No length penalty
            compression_ratio_threshold=2.4,  # Threshold for compression ratio
            no_speech_threshold=0.5,  # Lowered from 0.6 - more sensitive to speech
            cancellation_event=cancellation_event
        )
        
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Local FastWhisper STT cancelled after processing{Fore.RESET}")
            return None
        
        if result:
            logging.info(f"{Fore.GREEN}Local FastWhisper direct buffer transcription completed successfully{Fore.RESET}")
            return result
        else:
            logging.warning(f"{Fore.YELLOW}Local FastWhisper transcription returned no result{Fore.RESET}")
            return None
                
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info(f"{Fore.YELLOW}Local FastWhisper STT cancelled due to interruption{Fore.RESET}")
            return None
        logging.error(f"{Fore.RED}Local FastWhisper direct buffer transcription error: {e}{Fore.RESET}")
        raise