# 🎤 Voice Assistant - Local AI Voice to Voice Interface

A sophisticated, locally-running voice assistant that enables natural voice conversations with AI models through advanced speech recognition, language processing, and text-to-speech capabilities.

## 🚀 Project Motivation

In today's digital age, voice interaction has become the most natural and intuitive way to communicate with technology. This project was born from the vision to create a seamless, privacy-focused voice interface that brings AI conversations to life through natural speech.

**Our Core Vision:**
- **Natural Communication**: Transform how users interact with AI by enabling voice-to-voice conversations
- **Privacy First**: Keep all processing local, ensuring your conversations stay private
- **Seamless Experience**: Create an interface so natural that users forget they're talking to a machine
- **Accessibility**: Make AI technology accessible to everyone, regardless of typing ability or technical expertise

**Why Voice Matters:**
Voice interaction eliminates the barriers between human thought and AI response. It's faster than typing, more natural than clicking, and enables a truly conversational experience that feels like talking to a knowledgeable friend. Whether you're cooking, driving, or simply prefer speaking, this assistant adapts to your lifestyle.

## ✨ Key Features

### 🎯 **Voice Activity Detection (VAD) - Interruption Support**
- **Real-time Speech Detection**: Advanced VAD technology continuously monitors audio input
- **Instant Interruption**: Users can interrupt the AI's response at any time by simply starting to speak
- **Natural Conversation Flow**: Enables dynamic, human-like conversations without waiting for responses
- **Smart Audio Management**: Automatically handles audio playback interruption and resumption

### 🗣️ **Speech-to-Text (STT)**
- **Local Processing**: Powered by Faster Whisper for offline transcription
- **High Accuracy**: State-of-the-art speech recognition with minimal latency
- **Multi-language Support**: Handles various languages and accents
- **Real-time Streaming**: Processes audio as it's being spoken

### 🧠 **AI Language Model Integration**
- **Ollama Support**: Seamlessly integrates with local Ollama models
- **Multiple Model Options**: Support for various open-source LLMs (GPT-OSS, Llama, Mistral, etc.)
- **Local Processing**: All AI processing happens on your device
- **Customizable Responses**: Tailor the AI's personality and capabilities

### 🔊 **Text-to-Speech (TTS)**
- **Kokoro Voice Engine**: High-quality, natural-sounding voice synthesis
- **Local Processing**: No cloud dependencies for voice generation
- **Voice Customization**: Multiple voice options and customization parameters
- **Low Latency**: Fast response times for natural conversation flow

### 🎵 **Advanced Audio Management**
- **Multi-threaded Processing**: Efficient handling of concurrent audio operations
- **Process Management**: Intelligent management of AI and TTS processes
- **Audio Device Support**: Automatic detection and configuration of audio devices
- **Quality Optimization**: Optimized audio settings for clear communication

## 🏗️ Project Structure

```
voice-assistant/
├── 📁 voice_assistant/           # Core voice processing modules
│   ├── 🎤 audio.py              # Audio recording and playback
│   ├── 🗣️ transcription.py      # Speech-to-text processing
│   ├── 🧠 response_generation.py # AI response generation
│   ├── 🔊 text_to_speech.py     # Text-to-speech synthesis
│   ├── ⚙️ config.py             # Configuration management
│   ├── 🔑 api_key_manager.py    # API key management
│   ├── 🏠 local_stt.py          # Local STT implementation
│   ├── 🏠 local_tts.py          # Local TTS implementation
│   └── 🛠️ utils.py              # Utility functions
├── 🎯 vad_stt_llm_tts.py        # Main application file
├── 📊 process_manager.py         # Process management system
├── 🧵 thread_monitor.py          # Thread monitoring utilities
├── 📋 requirements.txt           # Python dependencies
├── 🎧 list_audio_devices.py     # Audio device detection
└── 📚 README.md                  # This file
```

## 🛠️ Setup Instructions

### Prerequisites

- **Python 3.10 or above** (Python 3.11+ recommended)
- **macOS 12 Monterey or later** (for macOS users)
- **Windows 10/11** (for Windows users)
- **Linux** (Ubuntu 20.04+ recommended)
- **16GB RAM minimum** (32GB+ recommended for optimal performance)
- **4 ~ 20 GB free disk space** for models and dependencies depends upon LLM

### Step 1: Install Ollama

1. **Download Ollama**: Visit [https://ollama.com/download](https://ollama.com/download)
2. **Install for your platform**:
   - **macOS**: Download the `.dmg` file and follow installation prompts
   - **Windows**: Download the `.exe` installer
   - **Linux**: Follow the curl installation command provided

3. **Verify Installation**: Open terminal/command prompt and run:
   ```bash
   ollama --version
   ```

### Step 2: Download AI Language Model

1. **Choose a Model**: Visit [https://ollama.com/search](https://ollama.com/search) to browse available models
2. **Download Model**: Run the appropriate command for your chosen model:

   **Popular Options:**
   ```bash
   # GPT-OSS (Recommended for reasoning and coding)
   ollama run gpt-oss
   
   # Llama 3.1 (Good balance of performance and speed)
   ollama run llama3.1
   
   # Mistral (Fast and efficient)
   ollama run mistral
   
   # DeepSeek (Excellent for coding tasks)
   ollama run deepseek-coder
   ```

3. **Wait for Download**: The first run will download the model (this may take several minutes depending on your internet speed)

### Step 3: Clone and Setup Project

1. **Clone Repository**:
   ```bash
   git clone https://github.com/HamzaRaouf/voice-assistant-ai.git
   cd voice-assistant
   ```

2. **Create Virtual Environment** (Recommended):
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Download TTS Model Files

The project uses Kokoro for high-quality text-to-speech. Download the required model files:

```bash
# Download Kokoro ONNX model
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx

# Download voice configuration file
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

**Note**: If you don't have `wget` installed:
- **macOS**: Install via Homebrew: `brew install wget`
- **Windows**: Use PowerShell: `Invoke-WebRequest -Uri "URL" -OutFile "filename"`
- **Linux**: Usually pre-installed

### Step 5: Configure Audio

1. **Check Audio Devices**:
   ```bash
   python list_audio_devices.py
   ```

2. **Update Configuration**: Edit `voice_assistant/config.py` with your preferred audio settings

### Step 6: Run the Application

```bash
python vad_stt_llm_tts.py
```

**First Run Notes**:
- The application will download additional models on first startup
- This may take 5-10 minutes depending on your internet connection
- Subsequent runs will be much faster

## 🎮 Usage

### Basic Operation

1. **Start the Application**: Run the main script
2. **Wait for Initialization**: Models will load (indicated by status messages)
3. **Speak Naturally**: Begin your conversation with the AI
4. **Interrupt When Needed**: Simply start speaking to interrupt the AI's response

### Voice Commands

- **Natural Conversation**: Ask questions, request help, or engage in casual chat
- **Task Assistance**: Get help with coding, writing, or problem-solving
- **Information Queries**: Ask about current events, science, or any topic

### Interruption Feature

The **Voice Activity Detection (VAD)** system enables natural conversation flow:
- **Automatic Detection**: System continuously monitors for speech input
- **Instant Response**: Interrupt AI responses by starting to speak
- **Seamless Transition**: No need to wait for AI to finish speaking

## ⚙️ Configuration

### Audio Settings

Edit `voice_assistant/config.py` to customize:
- **Sample Rate**: Audio quality and processing speed
- **Chunk Size**: Audio buffer management
- **Device Selection**: Input/output audio devices
- **VAD Sensitivity**: Speech detection sensitivity

### Model Selection

Choose different Ollama models based on your needs:
- **Performance**: Larger models (70B+) for complex reasoning
- **Speed**: Smaller models (7B-13B) for faster responses
- **Specialization**: Coding models for development tasks

## 🔧 Troubleshooting

### Common Issues

1. **Audio Not Working**:
   - Check microphone permissions
   - Verify audio device selection
   - Run `python list_audio_devices.py` to debug

2. **Model Download Issues**:
   - Ensure stable internet connection
   - Check available disk space
   - Verify Ollama installation

3. **Performance Issues**:
   - Close unnecessary applications
   - Ensure adequate RAM (8GB+)
   - Consider using smaller models

### Getting Help

- Check the [Issues](https://github.com/your-repo/issues) page
- Review [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md) for advanced features
- Consult [README_LOCAL_PROCESSING.md](README_LOCAL_PROCESSING.md) for local processing details

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Contribution Process

1. **Fork the Repository**: Click the "Fork" button on GitHub
2. **Create Feature Branch**: 
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make Changes**: Implement your improvements
4. **Commit Changes**:
   ```bash
   git commit -m 'Add amazing feature'
   ```
5. **Push to Branch**:
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open Pull Request**: Create a PR with detailed description

### Areas for Contribution

- **Audio Processing**: Improve VAD, noise reduction, audio quality
- **Model Integration**: Add support for more AI models
- **User Interface**: Enhance the user experience
- **Documentation**: Improve guides and examples
- **Testing**: Add comprehensive test coverage
- **Performance**: Optimize processing speed and memory usage

### Code Standards

- Follow PEP 8 Python style guidelines
- Add type hints where appropriate
- Include docstrings for new functions
- Write tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Voice Assistant Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- **Ollama Team**: For providing the local AI model infrastructure
- **Faster Whisper**: For efficient speech recognition
- **Kokoro Team**: For high-quality text-to-speech synthesis
- **Open Source Community**: For the amazing tools and libraries that make this possible

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/your-repo/issues)
- **Discussions**: [Join community discussions](https://github.com/your-repo/discussions)
- **Wiki**: [Check our documentation wiki](https://github.com/your-repo/wiki)

---

**Made with ❤️ by the Voice Assistant Community**

*Transform your conversations with AI through the power of voice.* 
