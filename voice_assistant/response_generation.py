# voice_assistant/response_generation.py

import logging
import threading
import time

from openai import OpenAI
from groq import Groq
# import ollama  # No longer needed - using OpenAI SDK for Ollama

from voice_assistant.config import Config

# Suppress HTTP request logs from various libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)  
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)


def generate_response(model:str, api_key:str, chat_history:list, local_model_path:str=None, cancellation_event=None):
    """
    Generate a response using the specified model.
    
    Args:
    model (str): The model to use for response generation ('openai', 'groq', 'local').
    api_key (str): The API key for the response generation service.
    chat_history (list): The chat history as a list of messages.
    local_model_path (str): The path to the local model (if applicable).
    cancellation_event (threading.Event): Event to check for cancellation

    Returns:
    str: The generated response text.
    """
    print("Model is : ",model)
    try:
        if model == 'openai':
            return _generate_openai_response(api_key, chat_history, cancellation_event)
        elif model == 'groq':
            return _generate_groq_response(api_key, chat_history, cancellation_event)
        elif model == 'ollama':
            return _generate_ollama_response(chat_history, cancellation_event)
        elif model == 'local':
            # Placeholder for local LLM response generation
            return "Generated response from local model"
        else:
            raise ValueError("Unsupported response generation model")
    except Exception as e:
        logging.error(f"Failed to generate response: {e}")
        return "Error in generating response"

def _generate_openai_response(api_key, chat_history, cancellation_event=None):
    client = OpenAI(api_key=api_key)
    
    # Check for cancellation before making API call
    if cancellation_event and cancellation_event.is_set():
        logging.info("OpenAI response generation cancelled before API call")
        return None
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_history
    )
    
    # Check for cancellation after API call
    if cancellation_event and cancellation_event.is_set():
        logging.info("OpenAI response generation cancelled after API call")
        return None

    return response.choices[0].message.content

def _generate_groq_response(api_key, chat_history, cancellation_event=None):
    client = Groq(api_key=api_key)
    
    # Check for cancellation before making API call
    if cancellation_event and cancellation_event.is_set():
        logging.info("Groq response generation cancelled before API call")
        return None
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=chat_history
    )
    
    # Check for cancellation after API call
    if cancellation_event and cancellation_event.is_set():
        logging.info("Groq response generation cancelled after API call")
        return None

    return response.choices[0].message.content


def _generate_ollama_response(chat_history, cancellation_event=None):
    """
    Generate Ollama response using OpenAI SDK with cancellation support and streaming
    """
    print("In Ollama Model (using OpenAI SDK)")
    
    # Check for cancellation before starting
    if cancellation_event and cancellation_event.is_set():
        logging.info("Ollama response generation cancelled before starting")
        return None
    
    try:
        # Create OpenAI client configured for Ollama
        client = OpenAI(
            base_url=Config.OLLAMA_BASE_URL,
            api_key='ollama'  # Ollama doesn't require a real API key
        )
        
        # Use streaming to allow for interruption during generation
        response_parts = []
        
        # Check for cancellation before API call
        if cancellation_event and cancellation_event.is_set():
            logging.info("Ollama response generation cancelled before API call")
            return None
        
        # Start streaming chat completion
        stream = client.chat.completions.create(
            model=Config.OLLAMA_LLM,
            messages=chat_history,
            temperature=Config.OLLAMA_TEMPERATURE,
            stream=True,
            max_tokens=1000
        )
        
        for chunk in stream:
            # Check for cancellation during streaming
            if cancellation_event and cancellation_event.is_set():
                logging.info("Ollama response generation cancelled during streaming")
                return None
            
            # Accumulate response parts
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                if content:
                    response_parts.append(content)
                    
            # Small delay to allow cancellation checks
            time.sleep(0.0001)
        
        # Final cancellation check
        if cancellation_event and cancellation_event.is_set():
            logging.info("Ollama response generation cancelled after completion")
            return None
        
        # Combine all parts
        full_response = ''.join(response_parts)
        logging.info(f"Ollama response generated successfully (length: {len(full_response)})")
        return full_response
        
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info("Ollama response generation cancelled due to interruption")
            return None
        
        # Try non-streaming fallback if streaming fails
        logging.warning(f"Ollama streaming failed: {e}, trying non-streaming fallback")
        try:
            # Check for cancellation before fallback
            if cancellation_event and cancellation_event.is_set():
                logging.info("Ollama response generation cancelled before fallback")
                return None
            
            # Create OpenAI client for non-streaming request
            client = OpenAI(
                base_url=Config.OLLAMA_BASE_URL,
                api_key='ollama'
            )
            
            response = client.chat.completions.create(
                model=Config.OLLAMA_LLM,
                messages=chat_history,
                temperature=Config.OLLAMA_TEMPERATURE
            )
            
            # Check for cancellation after API call
            if cancellation_event and cancellation_event.is_set():
                logging.info("Ollama response generation cancelled after fallback API call")
                return None
            
            fallback_response = response.choices[0].message.content
            logging.info(f"Ollama fallback response generated successfully (length: {len(fallback_response)})")
            return fallback_response
            
        except Exception as fallback_error:
            if cancellation_event and cancellation_event.is_set():
                logging.info("Ollama response generation cancelled during fallback")
                return None
            logging.error(f"Both streaming and non-streaming Ollama requests failed: {fallback_error}")
            raise fallback_error

def _generate_ollama_response_non_streaming(chat_history, cancellation_event=None):
    """
    Fallback non-streaming version for Ollama using OpenAI SDK
    """
    print("In Ollama Model (non-streaming with OpenAI SDK)")
    
    # Check for cancellation before starting
    if cancellation_event and cancellation_event.is_set():
        logging.info("Ollama non-streaming response generation cancelled before starting")
        return None
    
    try:
        # Create OpenAI client configured for Ollama
        client = OpenAI(
            base_url=Config.OLLAMA_BASE_URL,
            api_key='ollama'
        )
        
        response = client.chat.completions.create(
            model=Config.OLLAMA_LLM,
            messages=chat_history,
            temperature=Config.OLLAMA_TEMPERATURE
        )
        
        # Check for cancellation after completion
        if cancellation_event and cancellation_event.is_set():
            logging.info("Ollama non-streaming response generation cancelled after completion")
            return None
        
        result = response.choices[0].message.content
        logging.info(f"Ollama non-streaming response generated successfully (length: {len(result)})")
        return result
        
    except Exception as e:
        if cancellation_event and cancellation_event.is_set():
            logging.info("Ollama non-streaming response generation cancelled due to interruption")
            return None
        logging.error(f"Error in Ollama non-streaming response generation: {e}")
        raise e