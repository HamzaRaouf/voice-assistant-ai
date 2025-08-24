#!/usr/bin/env python3
"""
Audio Device Manager for Voice Assistant

This script provides utilities for managing audio devices in the voice assistant system.
It demonstrates the enhanced device detection and management capabilities.
"""

import os
import sys
import logging
from colorama import Fore, init

# Initialize colorama for cross-platform colored output
init()

# Add the project directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voice_assistant.audio import (
    get_available_input_devices, 
    auto_select_input_device, 
    list_audio_devices
)
from voice_assistant.config import Config

def main():
    """Main function to demonstrate audio device management"""
    print(f"{Fore.GREEN}🎤 Voice Assistant Audio Device Manager{Fore.RESET}")
    print("="*60)
    
    try:
        # List all available audio devices
        print(f"\n{Fore.CYAN}1. Available Audio Input Devices:{Fore.RESET}")
        list_audio_devices()
        
        # Show auto-selected device
        print(f"{Fore.CYAN}2. Auto-Selected Device:{Fore.RESET}")
        try:
            selected_index = auto_select_input_device()
            devices = get_available_input_devices()
            selected_device = next((d for d in devices if d['index'] == selected_index), None)
            if selected_device:
                print(f"   Auto-selected: {selected_device['name']} (Index: {selected_index})")
            else:
                print(f"   Auto-selected device index: {selected_index}")
        except Exception as e:
            print(f"   {Fore.RED}Error in auto-selection: {e}{Fore.RESET}")
        
        # Show current configuration
        print(f"\n{Fore.CYAN}3. Current Configuration:{Fore.RESET}")
        config_info = Config.get_local_config_info()
        
        print(f"   Preferred Device: {Config.PREFERRED_MICROPHONE_DEVICE or 'Auto-select'}")
        print(f"   Energy Threshold: {Config.ENERGY_THRESHOLD}")
        print(f"   Pause Threshold: {Config.PAUSE_THRESHOLD}s")
        print(f"   Phrase Threshold: {Config.PHRASE_THRESHOLD}s")
        print(f"   Calibration Duration: {Config.CALIBRATION_DURATION}s")
        print(f"   List Devices on Start: {Config.LIST_AUDIO_DEVICES_ON_START}")
        
        print(f"\n{Fore.CYAN}4. STT Configuration:{Fore.RESET}")
        print(f"   Model: {config_info['transcription']['model']}")
        print(f"   Whisper Model Size: {config_info['transcription']['whisper_model_size']}")
        print(f"   Whisper Device: {config_info['transcription']['whisper_device']}")
        print(f"   Language: {config_info['transcription']['whisper_language']}")
        print(f"   Audio Enhancement: {config_info['transcription']['audio_enhancement']}")
        
        # Configuration tips
        print(f"\n{Fore.YELLOW}💡 Configuration Tips:{Fore.RESET}")
        print("   To set a preferred microphone device:")
        print("   export PREFERRED_MICROPHONE_DEVICE=<device_index>")
        print("\n   To enable audio device listing on startup:")
        print("   export LIST_AUDIO_DEVICES_ON_START=true")
        print("\n   To configure Whisper settings:")
        print("   export WHISPER_MODEL_SIZE=medium")
        print("   export WHISPER_LANGUAGE=auto")
        print("   export WHISPER_ENABLE_AUDIO_ENHANCEMENT=true")
        
        print(f"\n{Fore.GREEN}✅ Audio device management complete!{Fore.RESET}")
        
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Fore.RESET}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    exit_code = main()
    sys.exit(exit_code) 