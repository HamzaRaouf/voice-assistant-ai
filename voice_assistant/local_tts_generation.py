import requests
import logging
import os
import time
import asyncio
from voice_assistant.config import Config
import sounddevice as sd
import numpy as np
# accent='EN-US'
import aiohttp
import aiofiles

async def cancel_kokoro_v2_tts():
    """
    Helper function to call the Kokoro V2 cancel API endpoint.
    This will stop any ongoing TTS generation and free resources.
    
    Returns:
        bool: True if cancellation was successful, False otherwise
    """
    try:
        cancel_url = "http://localhost:8500/cancel_tts"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(cancel_url) as response:
                if response.status == 200:
                    result = await response.json()
                    logging.info(f"Cancel API response: {result}")
                    return True
                else:
                    logging.warning(f"Cancel API failed with status: {response.status}")
                    return False
    except Exception as e:
        logging.error(f"Error calling cancel API: {e}")
        return False

async def ensure_kokoro_v2_cancelled_before_new_request():
    """
    Utility function to ensure any ongoing Kokoro V2 TTS is cancelled before making a new request.
    This helps prevent resource conflicts and ensures clean state.
    
    Returns:
        bool: True if cancellation was successful or no task was running, False otherwise
    """
    try:
        result = await cancel_kokoro_v2_tts()
        if result:
            # Give a small delay to ensure cancellation is processed
            await asyncio.sleep(0.1)
        return True
    except Exception as e:
        logging.error(f"Error ensuring cancellation before new request: {e}")
        return False

def generate_audio_file_melotts(text, language='EN', accent='EN-BR', speed=1.0, filename=None, cancellation_event=None):
    print("We are in Melo TTS :")
    """
    Generate an audio file from the given text using the FastAPI endpoint.

    Args:
        text (str): The text to convert to speech.
        language (str): The language of the text. Default is 'EN'.
        accent (str): The accent to use for the speech. Default is 'EN-US'.
        speed (float): The speed of the speech. Default is 1.0.
        filename (str, optional): The desired name for the output audio file. If None, a unique name will be generated.
        cancellation_event (threading.Event): Event to check for cancellation

    Returns:
        dict: A dictionary containing the message and the file path of the generated audio.
    """
    # Check for cancellation before starting
    if cancellation_event and cancellation_event.is_set():
        logging.info("MeloTTS generation cancelled before starting")
        return None
    
    # Define the API endpoint
    url = f"http://localhost:{Config.TTS_PORT_LOCAL}/generate-audio/"

    # Define the payload
    payload = {
        "text": text,
        "language": language,
        "accent": accent,
        "speed": speed
    }

    if filename:
        payload["filename"] = filename

    # Set the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Check for cancellation before making request
    if cancellation_event and cancellation_event.is_set():
        logging.info("MeloTTS generation cancelled before API call")
        return None

    try:
        # Make the POST request (simplified, no timeout for better stability like Verbi)
        response = requests.post(url, json=payload, headers=headers)
        
        # Check for cancellation after request
        if cancellation_event and cancellation_event.is_set():
            logging.info("MeloTTS generation cancelled after API call")
            return None

        # Check the response
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
            
    except requests.exceptions.RequestException as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info("MeloTTS generation cancelled due to interruption")
            return None
        raise e



async def generate_audio_file_kokoro_V2(text, language='EN', accent='af_jessica', speed=1.0, filename=None, cancellation_event=None):
    """
    Generate an audio file from the given text using the FastAPI endpoint.
    This version handles streaming properly and saves to file.

    Args:
        text (str): The text to convert to speech.
        language (str): The language of the text. Default is 'EN'.
        accent (str): The accent to use for the speech. Default is 'af_jessica'.
        speed (float): The speed of the speech. Default is 1.0.
        filename (str, optional): The desired name for the output audio file. If None, a unique name will be generated.
        cancellation_event (threading.Event): Event to check for cancellation

    Returns:
        str: The file path of the generated audio file.
    """
    # Check for cancellation before starting
    if cancellation_event and cancellation_event.is_set():
        logging.info("Kokoro V2 TTS generation cancelled before starting")
        # Call cancel API to ensure server resources are freed
        await cancel_kokoro_v2_tts()
        return None
    
    print("We are in Kokoro V2 TTS (Streaming):")
    
    # Ensure any previous TTS task is cancelled before starting new one
    await ensure_kokoro_v2_cancelled_before_new_request()
    
    # URL of your FastAPI endpoint
    # url = "https://6229e8a5268f.ngrok-free.app/tts"
    url = "http://127.0.0.1:8500/tts"
    # url = "http://localhost:8500/tts"

    # Create the payload - MUST match exactly what the server expects
    payload = {
        "text": text,
        "accent": "af_jessica",  # Server expects exactly this value
        "speed": 0.9,            # Server expects exactly this value
        "lang": "en-us"          # Server expects exactly this value
    }
    
    print(f"Sending payload to server: {payload}")

    try:
        # Use aiohttp for async HTTP requests
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                print(f"Server response status: {response.status}")
                print(f"Server response headers: {dict(response.headers)}")
                
                if response.status == 200:
                    print("Starting audio stream download...")
                    
                    # Read the entire streaming response
                    audio_data = await response.read()
                    print(f"Downloaded {len(audio_data)} bytes of audio data")
                    
                    # Check for cancellation after download
                    if cancellation_event and cancellation_event.is_set():
                        logging.info("Kokoro V2 TTS cancelled after download")
                        # Call cancel API to ensure server resources are freed
                        await cancel_kokoro_v2_tts()
                        return None
                    
                    # Save the audio data to file
                    if filename:
                        output_file = filename
                    else:
                        output_file = f"kokoro_v2_output_{int(time.time())}.wav"
                    
                    # Ensure the output file has .wav extension
                    if not output_file.endswith('.wav'):
                        output_file = output_file.replace('.mp3', '.wav')
                    
                    # Write the audio data to file
                    async with aiofiles.open(output_file, 'wb') as f:
                        await f.write(audio_data)
                    
                    print(f"Audio saved to: {output_file}")
                    return output_file
                else:
                    # Get error details from server
                    error_text = await response.text()
                    print(f"Failed to generate audio. Status code: {response.status}")
                    print(f"Error details: {error_text}")
                    return None
                    
    except Exception as e:
        print(f"Error in Kokoro V2 TTS: {e}")
        import traceback
        traceback.print_exc()
        # Call cancel API if there was an error and cancellation was requested
        if cancellation_event and cancellation_event.is_set():
            await cancel_kokoro_v2_tts()
        return None


async def generate_audio_file_kokoro_V2_streaming(text, api_key=None, local_model_path=None, language='EN', accent='af_jessica', speed=1.0, cancellation_event=None):
    """
    NEW: Generate streaming audio chunks from the given text using the FastAPI endpoint.
    This version yields audio chunks directly without saving to file for immediate playback.

    Args:
        text (str): The text to convert to speech.
        api_key (str): Not used for this implementation but kept for compatibility.
        local_model_path (str): Not used for this implementation but kept for compatibility.
        language (str): The language of the text. Default is 'EN'.
        accent (str): The accent to use for the speech. Default is 'af_jessica'.
        speed (float): The speed of the speech. Default is 1.0.
        cancellation_event (threading.Event): Event to check for cancellation

    Yields:
        bytes: Audio chunks for immediate playback
    """
    # Check for cancellation before starting
    if cancellation_event and cancellation_event.is_set():
        logging.info("Kokoro V2 streaming TTS generation cancelled before starting")
        # Call cancel API to ensure server resources are freed
        await cancel_kokoro_v2_tts()
        return
    
    print("We are in Kokoro V2 Streaming TTS (No file saving):")
    
    # Ensure any previous TTS task is cancelled before starting new one
    await ensure_kokoro_v2_cancelled_before_new_request()
    
    # URL of your FastAPI endpoint
    # url = "https://6229e8a5268f.ngrok-free.app/tts"
    # url = "http://192.168.100.244:8500/tts"
    url = "http://localhost:8500/tts"
    # url = "http://192.168.18.16:8500/tts"

    # Create the payload - MUST match exactly what the server expects
    payload = {
        "text": text,
        "accent": "af_jessica",  # Server expects exactly this value
        "speed": 0.9,            # Server expects exactly this value
        "lang": "en-us"          # Server expects exactly this value
    }
    
    print(f"Sending payload to server: {payload}")

    try:
        # Use aiohttp for async HTTP requests
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                print(f"Server response status: {response.status}")
                print(f"Server response headers: {dict(response.headers)}")
                
                if response.status == 200:
                    print("Starting streaming audio download...")
                    
                    # Stream the response in chunks instead of reading all at once
                    chunk_size = 4096  # 4KB chunks for smooth streaming
                    total_bytes = 0
                    
                    async for chunk in response.content.iter_chunked(chunk_size):
                        # Check for cancellation during streaming
                        if cancellation_event and cancellation_event.is_set():
                            logging.info("Kokoro V2 streaming TTS cancelled during download")
                            # Call cancel API to ensure server resources are freed
                            await cancel_kokoro_v2_tts()
                            return
                        
                        total_bytes += len(chunk)
                        print(f"Received chunk: {len(chunk)} bytes (total: {total_bytes} bytes)")
                        
                        # Yield the chunk for immediate playback
                        yield chunk
                    
                    print(f"Streaming completed. Total bytes: {total_bytes}")
                    
                else:
                    # Get error details from server
                    error_text = await response.text()
                    print(f"Failed to generate audio. Status code: {response.status}")
                    print(f"Error details: {error_text}")
                    return
                    
    except Exception as e:
        print(f"Error in Kokoro V2 Streaming TTS: {e}")
        import traceback
        traceback.print_exc()
        # Call cancel API if there was an error and cancellation was requested
        if cancellation_event and cancellation_event.is_set():
            await cancel_kokoro_v2_tts()
        return


def generate_audio_file_kokoro(text, language='EN', accent='af_jessica', speed=1.0, filename=None, cancellation_event=None):
    print("We are in Kokoro TTS :")
    """
    Generate an audio file from the given text using the Kokoro TTS model.
    """
    # Check for cancellation before starting
    if cancellation_event and cancellation_event.is_set():
        logging.info("Kokoro TTS generation cancelled before starting")
        return None

    # Define the URL to the remote TTS service
    # url = f"http://<NEW_SYSTEM_IP>:<PORT>/generate-audio/"  # Replace with the actual IP and port of the remote system
    url = f"https://7fe886ffa6c8.ngrok-free.app/generate-audio/"  # Replace with the actual IP and port of the remote system

    # Define the payload (what will be sent to the TTS service)
    payload = {
        "text": text,
        "language": language,
        "accent": accent,
        "speed": speed,
    
    }
    if filename:
        payload["filename"] = filename


    headers = {
        "Content-Type": "application/json"
    }

    # Check for cancellation before making the request
    if cancellation_event and cancellation_event.is_set():
        logging.info("Kokoro TTS generation cancelled before API call")
        return None

    print("Payload:", payload)

    try:
        # Make the request to the remote TTS service (Kokoro)
        response = requests.post(url, json=payload, headers=headers)

        # Check for cancellation after request
        if cancellation_event and cancellation_event.is_set():
            logging.info("Kokoro TTS generation cancelled after API call")
            return None

        # Handle the response from the TTS service
        if response.status_code == 200:
            print("Response content type:", response.headers['Content-Type'])
            print("Response content length:", len(response.content))
            with open("output.wav", "wb") as f:
                f.write(response.content)
            print("Audio file saved successfully.")
            return response.json().get('file_path', None)
        else:
            logging.error(f"TTS service failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during TTS request: {e}")
        if cancellation_event and cancellation_event.is_set():
            logging.info("Kokoro TTS generation cancelled due to interruption")
        return None
# def generate_audio_file_kokoro(text, language='EN', accent='af_jessica', speed=1.0, filename=None, cancellation_event=None):
    print("We are in Kokoro TTS :")
    """
    Generate an audio file from the given text using the Kokoro TTS model.
    """
    # Check for cancellation before starting
    if cancellation_event and cancellation_event.is_set():
        logging.info("Kokoro TTS generation cancelled before starting")
        return None

    url = "https://7fe886ffa6c8.ngrok-free.app/generate-audio/"  # FastAPI URL

    payload = {
        "text": text,
        "language": language,
        "accent": accent,
        "speed": speed,
    }
    if filename:
        payload["filename"] = filename

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)

        # Check if response is successful and content is valid
        if response.status_code == 200:
            print("Response content type:", response.headers['Content-Type'])
            print("Response content length:", len(response.content))

            # Write the audio data directly to a file
            with open("output.wav", "wb") as f:
                f.write(response.content)  # Write raw binary data

            print("Audio file saved successfully.",response.json().get('file_path', None))
            return response.json().get('file_path', None)
        else:
            logging.error(f"TTS service failed: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error during TTS request: {e}")
        return None



def generate_audio_file_cartesia(text, filename=None, cancellation_event=None):
    """
    Generate audio using Cartesia TTS with cancellation support
    """
    # Check for cancellation before starting
    if cancellation_event and cancellation_event.is_set():
        logging.info("Cartesia TTS generation cancelled before starting")
        return None
    
    try:
        from cartesia import Cartesia
        import pyaudio
        import soundfile as sf
        import numpy as np
        
        # Initialize Cartesia client
        client = Cartesia(api_key=Config.CARTESIA_API_KEY)
        voice_id = "f114a467-c40a-4db8-964d-aaba89cd08fa"
        voice = client.voices.get(id=voice_id)
        
        model_id = "sonic-english"
        output_format = {
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": 44100,
        }
        
        # Check for cancellation before generation
        if cancellation_event and cancellation_event.is_set():
            logging.info("Cartesia TTS generation cancelled before generation")
            return None
        
        audio_data = []
        
        # Generate and collect audio with cancellation checks
        for output in client.tts.sse(
            model_id=model_id,
            transcript=text,
            voice_embedding=voice["embedding"],
            stream=True,
            output_format=output_format,
        ):
            # Check for cancellation during streaming
            if cancellation_event and cancellation_event.is_set():
                logging.info("Cartesia TTS generation cancelled during streaming")
                return None
            
            buffer = output["audio"]
            audio_data.append(np.frombuffer(buffer, dtype=np.float32))
            
            # Small delay to allow cancellation checks
            time.sleep(0.001)
        
        # Final cancellation check
        if cancellation_event and cancellation_event.is_set():
            logging.info("Cartesia TTS generation cancelled after generation")
            return None
        
        # Combine and save audio
        if audio_data and filename:
            combined_audio = np.concatenate(audio_data)
            sf.write(filename, combined_audio, 44100)
        
        return filename
        
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info("Cartesia TTS generation cancelled due to interruption")
            return None
        raise e

def generate_audio_file_piper(text, filename=None, cancellation_event=None):
    """
    Generate audio using Piper TTS with cancellation support
    """
    # Check for cancellation before starting
    if cancellation_event and cancellation_event.is_set():
        logging.info("Piper TTS generation cancelled before starting")
        return None
    
    try:
        # Check for cancellation before API call
        if cancellation_event and cancellation_event.is_set():
            logging.info("Piper TTS generation cancelled before API call")
            return None
        
        response = requests.post(
            f"{Config.PIPER_SERVER_URL}/synthesize/",
            json={"text": text},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Check for cancellation after API call
        if cancellation_event and cancellation_event.is_set():
            logging.info("Piper TTS generation cancelled after API call")
            return None
        
        if response.status_code == 200:
            output_file = filename or Config.PIPER_OUTPUT_FILE
            with open(output_file, "wb") as f:
                f.write(response.content)
            logging.info(f"Piper TTS output saved to {output_file}")
            return output_file
        else:
            logging.error(f"Piper TTS API error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info("Piper TTS generation cancelled due to interruption")
            return None
        raise e

# Example usage of the function
if __name__ == "__main__":
    try:
        text = "Hello, this is a test of the TTS system."
        result = generate_audio_file_melotts(text, filename="test_output.mp3")
        print(f"Audio generation result: {result}")
    except Exception as e:
        print(f"Error: {e}")