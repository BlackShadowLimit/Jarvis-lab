# Jarvis lab

Drawing inspiration from Iron Man's J.A.R.V.I.S., this project aims to develop a local virtual assistant for MacOS.  The assistant will have the ability to access and interpret screen and file content.  Given the constraints of Apple M-series GPUs for LLM workloads, the design prioritizes efficient speech-to-text and text-to-speech pipelines, along with memory optimization.

## Speech to text (STT)

I'm working on a project that uses a [local Whisper model](https://github.com/openai/whisper) and PyAudio to achieve real-time audio transcription.



## Text to speech (TTS)

For text-to-speech, I'm using [Conqui TTS](https://github.com/coqui-ai/TTS).  It has a longer initial startup time than pyttsx3, but the trade-off is much faster audio synthesis, which is more important for my use case.



## LLM (Brain)

For the virtual assistant's LLM, I've selected [Llama 3.2 3b](https://ollama.com/library/llama3.2:3b).  This model's resource efficiency enables execution on 8GB RAM configurations.  The memory system consists of short-term memory, implemented as a JSON file storing the last 10 interactions, and long-term memory, which uses FAISS for vector-based compression of the complete conversation history.