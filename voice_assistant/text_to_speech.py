# voice_assistant/text_to_speech.py

import logging
import threading
import time
import requests
import elevenlabs
from elevenlabs import ElevenLabs
from openai import OpenAI
from deepgram import DeepgramClient, SpeakOptions
from .local_tts_generation import generate_audio_file_melotts, generate_audio_file_cartesia, generate_audio_file_piper, generate_audio_file_kokoro,generate_audio_file_kokoro_V2
from .config import Config
import asyncio
import aiohttp
import aiofiles
from io import BytesIO
import tempfile
import os

# Suppress HTTP request logs from various libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)
logging.getLogger('elevenlabs').setLevel(logging.WARNING)
logging.getLogger('deepgram').setLevel(logging.WARNING)

# Import the new local TTS engine
from .local_tts import get_local_tts_engine, generate_speech_local, generate_speech_stream_local

async def text_to_speech_buffer(model: str, api_key: str, text: str, local_model_path: str = None, cancellation_event=None):
    """Generate TTS using the specified model and return audio buffer with cancellation support"""
    try:
        if cancellation_event and cancellation_event.is_set():
            return None
            
        if model == "openai":
            return await _generate_openai_tts_buffer(api_key, text, cancellation_event)
        elif model == "deepgram":
            return await _generate_deepgram_tts_buffer(api_key, text, cancellation_event)
        elif model == "elevenlabs":
            return await _generate_elevenlabs_tts_buffer(api_key, text, cancellation_event)
        elif model == "cartesia":
            return await _generate_cartesia_tts_buffer(api_key, text, cancellation_event)
        elif model == "melotts":
            return await _generate_melotts_tts_buffer(text, cancellation_event)
        elif model == "kokoro":
            return await _generate_kokoro_tts_buffer(text, cancellation_event)
        elif model == "kokoro_V2":
            return await _generate_kokoro_tts_V2_buffer(text, cancellation_event)
        elif model == "kokoro_local":
            # NEW: Use local Kokoro TTS directly without HTTP
            return await _generate_kokoro_tts_local_buffer(text, cancellation_event)
        elif model == "piper":
            return await _generate_piper_tts_buffer(text, cancellation_event)
        elif model == 'local':
            # Placeholder for local TTS generation
            if cancellation_event and cancellation_event.is_set():
                return None
            return BytesIO(b"Generated speech from local model")
        else:
            raise ValueError("Unsupported TTS model")
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info("TTS generation cancelled due to interruption")
            return None
        logging.error(f"Failed to generate speech: {e}")
        return None

async def _generate_kokoro_tts_local_buffer(text, cancellation_event=None):
    """
    NEW: Local Kokoro TTS generation that returns audio buffer without saving to file.
    """
    try:
        if cancellation_event and cancellation_event.is_set():
            logging.info("Local Kokoro TTS cancelled before starting")
            return None
        
        # Use the local TTS engine for buffer generation
        engine = get_local_tts_engine()
        
        audio_buffer = await engine.generate_audio_buffer(
            text=text,
            voice="af_jessica",  # Default voice
            speed=0.9,  # Default speed
            lang="en-us",  # Default language
            cancellation_event=cancellation_event
        )
        
        if cancellation_event and cancellation_event.is_set():
            logging.info("Local Kokoro TTS cancelled after generation")
            return None
        
        return audio_buffer
            
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info("Local Kokoro TTS cancelled due to interruption")
            return None
        logging.error(f"Local Kokoro TTS generation error: {e}")
        raise

async def _generate_openai_tts_buffer(api_key, text, cancellation_event=None):
    """Generate TTS using OpenAI and return audio buffer"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    client = OpenAI(api_key=api_key)
    speech_response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )
    
    if cancellation_event and cancellation_event.is_set():
        logging.info("OpenAI TTS cancelled after generation")
        return None
    
    # Convert response to BytesIO buffer
    audio_buffer = BytesIO()
    for chunk in speech_response.iter_bytes():
        audio_buffer.write(chunk)
    audio_buffer.seek(0)
    return audio_buffer

async def _generate_deepgram_tts_buffer(api_key, text, cancellation_event=None):
    """Generate TTS using Deepgram and return audio buffer"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    try:
        deepgram = DeepgramClient(api_key)
        options = SpeakOptions(
            model="aura-luna-en",
            encoding="mp3",
            container="mp3"
        )
        
        speak_result = deepgram.speak.v("1").stream({"text": text}, options)
        
        if cancellation_event and cancellation_event.is_set():
            logging.info("Deepgram TTS cancelled after generation")
            return None
        
        # Convert response to BytesIO buffer
        audio_buffer = BytesIO()
        for chunk in speak_result.iter_content():
            audio_buffer.write(chunk)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            return None
        raise e

async def _generate_elevenlabs_tts_buffer(api_key, text, cancellation_event=None):
    """Generate TTS using ElevenLabs and return audio buffer"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    client = ElevenLabs(api_key=api_key)
    
    if cancellation_event and cancellation_event.is_set():
        logging.info("ElevenLabs TTS cancelled before generation")
        return None
        
    audio = client.generate(
        text=text, 
        voice="Paul J.", 
        output_format="mp3_22050_32", 
        model="eleven_turbo_v2"
    )
    
    if cancellation_event and cancellation_event.is_set():
        logging.info("ElevenLabs TTS cancelled after generation")
        return None
    
    # Convert generator to buffer
    audio_buffer = BytesIO()
    for chunk in audio:
        audio_buffer.write(chunk)
    audio_buffer.seek(0)
    return audio_buffer

# For the other TTS services that don't have direct buffer support,
# we'll need to use temporary files and then read them into buffers
async def _generate_cartesia_tts_buffer(api_key, text, cancellation_event=None):
    """Generate TTS using Cartesia via temp file and return audio buffer"""
    if cancellation_event and cancellation_event.is_set():
        return None
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Use existing file-based function 
        result = await _generate_cartesia_tts(api_key, text, temp_path, cancellation_event)
        
        if cancellation_event and cancellation_event.is_set():
            return None
        
        if result:
            # Read file into buffer
            with open(temp_path, 'rb') as f:
                audio_buffer = BytesIO(f.read())
            return audio_buffer
        return None
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

async def _generate_melotts_tts_buffer(text, cancellation_event=None):
    """Generate TTS using MeloTTS via temp file and return audio buffer"""
    if cancellation_event and cancellation_event.is_set():
        return None
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Use existing file-based function 
        result = _generate_melotts_tts(text, temp_path, cancellation_event)
        
        if cancellation_event and cancellation_event.is_set():
            return None
        
        if result:
            # Read file into buffer
            with open(temp_path, 'rb') as f:
                audio_buffer = BytesIO(f.read())
            return audio_buffer
        return None
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

async def _generate_kokoro_tts_buffer(text, cancellation_event=None):
    """Generate TTS using Kokoro via temp file and return audio buffer"""
    if cancellation_event and cancellation_event.is_set():
        return None
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Use existing file-based function 
        result = _generate_kokoro_tts(text, temp_path, cancellation_event)
        
        if cancellation_event and cancellation_event.is_set():
            return None
        
        if result:
            # Read file into buffer
            with open(temp_path, 'rb') as f:
                audio_buffer = BytesIO(f.read())
            return audio_buffer
        return None
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

async def _generate_kokoro_tts_V2_buffer(text, cancellation_event=None):
    """Generate TTS using Kokoro V2 via temp file and return audio buffer"""
    if cancellation_event and cancellation_event.is_set():
        return None
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Use existing file-based function 
        result = await _generate_kokoro_tts_V2(text, temp_path, cancellation_event)
        
        if cancellation_event and cancellation_event.is_set():
            return None
        
        if result:
            # Read file into buffer
            with open(temp_path, 'rb') as f:
                audio_buffer = BytesIO(f.read())
            return audio_buffer
        return None
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

async def _generate_piper_tts_buffer(text, cancellation_event=None):
    """Generate TTS using Piper via temp file and return audio buffer"""
    if cancellation_event and cancellation_event.is_set():
        return None
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Use existing file-based function 
        result = _generate_piper_tts(text, temp_path, cancellation_event)
        
        if cancellation_event and cancellation_event.is_set():
            return None
        
        if result:
            # Read file into buffer
            with open(temp_path, 'rb') as f:
                audio_buffer = BytesIO(f.read())
            return audio_buffer
        return None
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

async def text_to_speech(model: str, api_key:str, text:str, output_file_path:str, local_model_path:str=None, cancellation_event=None):
    """Generate TTS using the specified model with cancellation support"""
    try:
        if cancellation_event and cancellation_event.is_set():
            return None
            
        if model == "openai":
            return _generate_openai_tts(api_key, text, output_file_path, cancellation_event)
        elif model == "deepgram":
            return _generate_deepgram_tts(api_key, text, output_file_path, cancellation_event)
        elif model == "elevenlabs":
            return _generate_elevenlabs_tts(api_key, text, output_file_path, cancellation_event)
        elif model == "cartesia":
            return _generate_cartesia_tts(api_key, text, output_file_path, cancellation_event)
        elif model == "melotts":
            return _generate_melotts_tts(text, output_file_path, cancellation_event)
        elif model == "kokoro":
            return _generate_kokoro_tts(text, output_file_path, cancellation_event)
        elif model == "kokoro_V2":
            # kokoro_V2 is streaming and doesn't save to file, so we handle it differently
            return await _generate_kokoro_tts_V2(text, output_file_path, cancellation_event)
        elif model == "kokoro_local":
            # NEW: Use local Kokoro TTS directly without HTTP
            return await _generate_kokoro_tts_local(text, output_file_path, cancellation_event)
        elif model == "piper":
            return _generate_piper_tts(text, output_file_path, cancellation_event)
        elif model == 'local':
            # Placeholder for local TTS generation
            if cancellation_event and cancellation_event.is_set():
                return None
            return "Generated speech from local model"
        else:
            raise ValueError("Unsupported TTS model")
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info("TTS generation cancelled due to interruption")
            return None
        logging.error(f"Failed to generate speech: {e}")
        return None

async def _generate_kokoro_tts_local(text, output_file_path, cancellation_event=None):
    """
    NEW: Local Kokoro TTS generation without HTTP calls.
    This uses the local TTS engine directly for better performance and no network dependency.
    """
    try:
        if cancellation_event and cancellation_event.is_set():
            logging.info("Local Kokoro TTS cancelled before starting")
            return None
        
        logging.info(f"Using local Kokoro TTS for generation: {output_file_path}")
        
        # Use the local TTS engine with appropriate parameters
        result = await generate_speech_local(
            text=text,
            output_path=output_file_path,
            voice="af_jessica",
            speed=0.9,
            lang="en-us",
            cancellation_event=cancellation_event
        )
        
        if cancellation_event and cancellation_event.is_set():
            logging.info("Local Kokoro TTS cancelled after processing")
            return None
        
        if result:
            logging.info(f"Local Kokoro TTS generation completed successfully: {result}")
            return result
        else:
            logging.warning("Local Kokoro TTS generation returned no result")
            return None
            
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info("Local Kokoro TTS cancelled due to interruption")
            return None
        logging.error(f"Local Kokoro TTS generation error: {e}")
        raise

def _generate_openai_tts(api_key, text, output_file_path, cancellation_event=None):
    """Generate TTS using OpenAI with cancellation support"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    client = OpenAI(api_key=api_key)
    speech_response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )
    
    if cancellation_event and cancellation_event.is_set():
        logging.info("OpenAI TTS cancelled after generation")
        return None

    speech_response.stream_to_file(output_file_path)
    return output_file_path

def _generate_deepgram_tts(api_key, text, output_file_path, cancellation_event=None):
    """Generate TTS using Deepgram with cancellation support"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    client = DeepgramClient(api_key=api_key)
    options = SpeakOptions(
        model="aura-arcas-en", #"aura-luna-en", # https://developers.deepgram.com/docs/tts-models
        encoding="linear16",
        container="wav"
    )
    SPEAK_OPTIONS = {"text": text}
    
    if cancellation_event and cancellation_event.is_set():
        logging.info("Deepgram TTS cancelled before API call")
        return None
        
    response = client.speak.v("1").save(output_file_path, SPEAK_OPTIONS, options)
    
    if cancellation_event and cancellation_event.is_set():
        logging.info("Deepgram TTS cancelled after generation")
        return None
        
    return output_file_path

def _generate_elevenlabs_tts(api_key, text, output_file_path, cancellation_event=None):
    """Generate TTS using ElevenLabs with cancellation support"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    client = ElevenLabs(api_key=api_key)
    
    if cancellation_event and cancellation_event.is_set():
        logging.info("ElevenLabs TTS cancelled before generation")
        return None
        
    audio = client.generate(
        text=text, 
        voice="Paul J.", 
        output_format="mp3_22050_32", 
        model="eleven_turbo_v2"
    )
    
    if cancellation_event and cancellation_event.is_set():
        logging.info("ElevenLabs TTS cancelled after generation")
        return None
        
    elevenlabs.save(audio, output_file_path)
    return output_file_path

def _generate_cartesia_tts(api_key, text, output_file_path, cancellation_event=None):
    """Generate TTS using Cartesia with cancellation support"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    try:
        generate_audio_file_cartesia(text, filename=output_file_path, cancellation_event=cancellation_event)
        
        if cancellation_event and cancellation_event.is_set():
            logging.info("Cartesia TTS cancelled after generation")
            return None
            
        return output_file_path
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            return None
        raise e

def _generate_melotts_tts(text, output_file_path, cancellation_event=None):
    """Generate TTS using MeloTTS with cancellation support"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    try:
        # Use the local MeloTTS generation with cancellation support
        result = generate_audio_file_melotts(
            text, 
            filename=output_file_path, 
            cancellation_event=cancellation_event
        )
        
        if cancellation_event and cancellation_event.is_set():
            logging.info("MeloTTS cancelled after generation")
            return None
            
        return output_file_path if result else None
        
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            return None
        raise e


def _generate_kokoro_tts(text, output_file_path, cancellation_event=None):
    """Generate TTS using Kokoro with cancellation support (HTTP-based - deprecated)"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    try:
        # Use the Kokoro TTS model to generate audio
        result = generate_audio_file_kokoro(
            text, 
            filename=output_file_path, 
            cancellation_event=cancellation_event
        )
        
        if cancellation_event and cancellation_event.is_set():
            logging.info("Kokoro TTS cancelled after generation")
            return None
            
        return output_file_path if result else None
        
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            return None
        raise e

async def _generate_kokoro_tts_V2(text, output_file_path, cancellation_event=None):
    """Generate TTS using Kokoro V2 with cancellation support (HTTP-based - deprecated)"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    try:
        # Use the Kokoro V2 TTS model to generate audio (async)
        result = await generate_audio_file_kokoro_V2(
            text, 
            filename=output_file_path, 
            cancellation_event=cancellation_event
        )
        
        if cancellation_event and cancellation_event.is_set():
            logging.info("Kokoro V2 TTS cancelled after generation")
            return None
            
        return result  # Return the actual result from the async function
        
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            return None
        raise e


def _generate_piper_tts(text, output_file_path, cancellation_event=None):
    """Generate TTS using Piper with cancellation support"""
    if cancellation_event and cancellation_event.is_set():
        return None
        
    try:
        generate_audio_file_piper(text, filename=output_file_path, cancellation_event=cancellation_event)
        
        if cancellation_event and cancellation_event.is_set():
            logging.info("Piper TTS cancelled after generation")
            return None
            
        return output_file_path
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            return None
        raise e