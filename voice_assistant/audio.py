# # voice_assistant/audio.py

import speech_recognition as sr
import pygame
import time
import logging
import pydub
from io import BytesIO
from pydub import AudioSegment
from functools import lru_cache
import sounddevice as sd
import soundfile as sf
import numpy as np
import wave
import pyaudio
import logging
from colorama import Fore
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@lru_cache(maxsize=None)
def get_recognizer():
    """
    Return a cached speech recognizer instance
    """
    return sr.Recognizer()

def get_available_input_devices() -> List[Dict]:
    """
    Get list of available audio input devices with detailed information.
    
    Returns:
        List[Dict]: List of available input devices with their properties
    """
    devices = []
    audio = pyaudio.PyAudio()
    
    try:
        for i in range(audio.get_device_count()):
            device_info = audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': device_info['defaultSampleRate'],
                    'is_default': False
                })
    except Exception as e:
        logging.error(f"{Fore.RED}Error getting audio devices: {e}{Fore.RESET}")
    finally:
        audio.terminate()
    
    # Mark default device
    try:
        audio = pyaudio.PyAudio()
        default_device = audio.get_default_input_device_info()
        for device in devices:
            if device['index'] == default_device['index']:
                device['is_default'] = True
                break
        audio.terminate()
    except:
        pass
    
    return devices

def auto_select_input_device(preferred_device_index: Optional[int] = None) -> int:
    """
    Auto-select the best available input device.
    
    Args:
        preferred_device_index (int, optional): Preferred device index to try first
    
    Returns:
        int: Selected device index
    """
    devices = get_available_input_devices()
    if not devices:
        raise Exception("No audio input devices found!")
    
    # Use manually specified device if provided and valid
    if preferred_device_index is not None:
        device_indices = [d['index'] for d in devices]
        if preferred_device_index in device_indices:
            selected_device = next(d for d in devices if d['index'] == preferred_device_index)
            logging.info(f"{Fore.GREEN}Using specified device: {selected_device['name']}{Fore.RESET}")
            return preferred_device_index
        else:
            logging.warning(f"{Fore.YELLOW}Warning: Device index {preferred_device_index} not found, using auto-selection{Fore.RESET}")
    
    # Try to find the default input device first
    try:
        audio = pyaudio.PyAudio()
        default_device = audio.get_default_input_device_info()
        selected_device_index = default_device['index']
        selected_device = next(d for d in devices if d['index'] == selected_device_index)
        logging.info(f"{Fore.GREEN}Using default device: {selected_device['name']}{Fore.RESET}")
        audio.terminate()
        return selected_device_index
    except:
        # If no default, use the first available device
        selected_device = devices[0]
        logging.info(f"{Fore.GREEN}Using first available device: {selected_device['name']}{Fore.RESET}")
        return selected_device['index']

def list_audio_devices():
    """Print all available audio input devices for user reference."""
    print(f"\n{Fore.CYAN}🎧 Available Audio Input Devices:{Fore.RESET}")
    print("="*50)
    
    devices = get_available_input_devices()
    for device in devices:
        default_marker = " [DEFAULT]" if device['is_default'] else ""
        print(f"  {device['index']}: {device['name']}{default_marker}")
        print(f"     - Channels: {device['channels']}")
        print(f"     - Sample Rate: {device['sample_rate']:.0f} Hz")
        print()

def record_audio_to_memory(timeout=10, phrase_time_limit=None, retries=3, energy_threshold=1000, 
                          pause_threshold=1.8, phrase_threshold=0.3, dynamic_energy_threshold=True, 
                          calibration_duration=0.6, device_index=None):
    """
    Record audio from the microphone and return it as audio data in memory.
    Intelligent recording: Supports 20+ second sentences but responds quickly when user stops speaking.
    
    Args:
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_limit (int): Maximum time for the phrase to be recorded (in seconds).
    retries (int): Number of retries if recording fails.
    energy_threshold (int): Energy threshold for considering whether a given chunk of audio is speech or not.
    pause_threshold (float): How much silence the recognizer interprets as the end of a phrase (in seconds).
    phrase_threshold (float): Minimum length of a phrase to consider for recording (in seconds).
    dynamic_energy_threshold (bool): Whether to enable dynamic energy threshold adjustment.
    calibration_duration (float): Duration of the ambient noise calibration (in seconds).
    device_index (int, optional): Specific microphone device index to use (None for auto-selection)
    
    Returns:
    BytesIO: Audio data as MP3 format in memory, or None if recording failed
    """
    recognizer = get_recognizer()
    recognizer.energy_threshold = energy_threshold
    recognizer.pause_threshold = pause_threshold
    recognizer.phrase_threshold = phrase_threshold
    recognizer.dynamic_energy_threshold = dynamic_energy_threshold
    
    selected_device_index = None
    if device_index is not None:
        selected_device_index = device_index
        logging.info(f"{Fore.GREEN}Using specified microphone device index: {device_index}{Fore.RESET}")
    else:
        try:
            selected_device_index = auto_select_input_device()
        except Exception as e:
            logging.warning(f"{Fore.YELLOW}Could not auto-select device: {e}, using default{Fore.RESET}")
    
    for attempt in range(retries):
        try:
            # Use selected device or default
            mic_kwargs = {}
            if selected_device_index is not None:
                mic_kwargs['device_index'] = selected_device_index
            
            with sr.Microphone(**mic_kwargs) as source:
                # Reduced calibration time for faster response
                logging.info("Calibrating for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=calibration_duration)
                logging.info("Recording started")
                # Listen for the first phrase and extract it into audio data
                # Increased timeout to allow longer messages
                audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                logging.info("Recording complete")

                # Convert the recorded audio data to MP3 format in memory
                wav_data = audio_data.get_wav_data()
                audio_segment = pydub.AudioSegment.from_wav(BytesIO(wav_data))
                # Use Verbi's exact settings: 22kHz, mono, 128k (optimized for MeloTTS)
                mp3_buffer = BytesIO()
                audio_segment.export(mp3_buffer, format="mp3", bitrate="128k", parameters=["-ar", "22050", "-ac", "1"])
                mp3_buffer.seek(0)  # Reset buffer position to beginning
                return mp3_buffer
        except sr.WaitTimeoutError:
            logging.warning(f"Listening timed out, retrying... ({attempt + 1}/{retries})")
        except Exception as e:
            logging.error(f"Failed to record audio: {e}")
            if attempt == retries -1:
                raise
        
    logging.error("Recording failed after all retries")
    return None

def record_audio(file_path, timeout=10, phrase_time_limit=None, retries=3, energy_threshold=1000, 
                 pause_threshold=1.8, phrase_threshold=0.3, dynamic_energy_threshold=True, 
                 calibration_duration=0.6, device_index=None):
    """
    Record audio from the microphone and save it as an MP3 file.
    This function now uses record_audio_to_memory internally and saves the result.
    
    Args:
    file_path (str): The path to save the recorded audio file.
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_limit (int): Maximum time for the phrase to be recorded (in seconds).
    retries (int): Number of retries if recording fails.
    energy_threshold (int): Energy threshold for considering whether a given chunk of audio is speech or not.
    pause_threshold (float): How much silence the recognizer interprets as the end of a phrase (in seconds).
    phrase_threshold (float): Minimum length of a phrase to consider for recording (in seconds).
    dynamic_energy_threshold (bool): Whether to enable dynamic energy threshold adjustment.
    calibration_duration (float): Duration of the ambient noise calibration (in seconds).
    device_index (int, optional): Specific microphone device index to use (None for auto-selection)
    """
    audio_buffer = record_audio_to_memory(
        timeout=timeout,
        phrase_time_limit=phrase_time_limit,
        retries=retries,
        energy_threshold=energy_threshold,
        pause_threshold=pause_threshold,
        phrase_threshold=phrase_threshold,
        dynamic_energy_threshold=dynamic_energy_threshold,
        calibration_duration=calibration_duration,
        device_index=device_index
    )
    
    if audio_buffer:
        # Save the buffer to file
        with open(file_path, 'wb') as f:
            f.write(audio_buffer.getvalue())
        logging.info(f"Audio saved to: {file_path}")
    else:
        raise Exception("Failed to record audio")

def play_audio(file_path):
    """
    Play an audio file using pygame.
    
    Args:
    file_path (str): The path to the audio file to play.
    """
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except pygame.error as e:
        logging.error(f"Failed to play audio: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while playing audio: {e}")
    finally:
        pygame.mixer.quit()

def play_audio_from_buffer(audio_buffer):
    """
    Play audio directly from a BytesIO buffer using pygame.
    
    Args:
    audio_buffer (BytesIO): Audio data buffer to play.
    """
    try:
        # Reset buffer position to beginning
        audio_buffer.seek(0)
        
        pygame.mixer.init()
        pygame.mixer.music.load(audio_buffer)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    except pygame.error as e:
        logging.error(f"Failed to play audio from buffer: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while playing audio from buffer: {e}")
    finally:
        pygame.mixer.quit()

def play_audio_from_buffer_interruptible(audio_buffer, interruption_event=None):
    """
    Play audio directly from a BytesIO buffer using pygame with interruption support.
    
    Args:
    audio_buffer (BytesIO): Audio data buffer to play.
    interruption_event (threading.Event): Event to signal interruption
    
    Returns:
    bool: True if playback completed, False if interrupted
    """
    try:
        # Reset buffer position to beginning
        audio_buffer.seek(0)
        
        pygame.mixer.init()
        pygame.mixer.music.load(audio_buffer)
        pygame.mixer.music.play()
        
        # Check for interruption every 50ms instead of 100ms for faster response
        while pygame.mixer.music.get_busy():
            if interruption_event and interruption_event.is_set():
                logging.info(f"{Fore.YELLOW}Audio playback interrupted by speech detection{Fore.RESET}")
                pygame.mixer.music.stop()
                return False
            pygame.time.wait(50)  # Shorter wait for faster interruption response
        
        logging.info(f"{Fore.GREEN}Audio playback completed normally{Fore.RESET}")
        return True
        
    except pygame.error as e:
        logging.error(f"Failed to play audio from buffer: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while playing audio from buffer: {e}")
        return False
    finally:
        pygame.mixer.quit()




