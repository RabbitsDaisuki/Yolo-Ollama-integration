import ollama
from scipy.io.wavfile import write
from pathlib import Path
import numpy as np
import whisper
import pyttsx3
import sounddevice as sd
from src import config
import cv2
import logging


class LLMDetection():
    def __init__(self):
        print(f"{'-'*5} System Initializing {'-'* 5}")

        print("Loading TTS Engine...")
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        print(f"Loading Whisper Model ({str(config.WHISPER_MODEL_NAME)})...")
        # --- Model Setting ---
        self.ollama_model = config.OLLAMA_MODEL_NAME
        self.whisper_model = whisper.load_model(str(config.WHISPER_MODEL_NAME))
        config.ASK_HISTORY = [
            {
                'role': 'system', 
                # 'content':'You are a helpful voice assistant. Keep your answers short and concise, suitable for speech.'
                'content': 'You are a helpful assistant with eyes. You will receive objects detected by YOLO. Use this visual information to answer user questions.'
            }
        ]
        print(f"{'-'* 3} System Ready {'-'* 3}")    
    def _list_input_Devices(self):
        devices = sd.query_devices()
        # print(f"Debug - Type: {type(devices)}")
        # print(f"Debug - Content: {devices}")
        for sound_index, sound_device_info in enumerate(devices):
            if sound_device_info["max_input_channels"] > 0:
                print(f"{sound_index}. {sound_device_info['name']}")

    def _Take_user_sound(self):
        # my_recording = sd.rec(int(self.duration * self.sample_rate), samplerate = self.sample_rate, channels = 1)
        # sd.wait()
        # print("Recording finished.")
        # write(self.full_wav_path, self.sample_rate, my_recording)
        # return str(self.full_wav_path)
        audio_frames = []
        silent_chunks = 0
        has_started = False

        chunks_per_sound = config.SAMPLE_RATE / config.CHUNK_SIZE
        silence_limit_chunks = int(config.SILENCE_LIMIT * chunks_per_sound)

        with sd.InputStream(samplerate = config.SAMPLE_RATE, channels = 1, dtype = 'int16') as stream:
            while True:
                data, overflow = stream.read(int(config.CHUNK_SIZE))

                audio_chunk = data.flatten()
                rms = np.sqrt(np.mean(audio_chunk.astype(float)**2))

                if rms > config.THRESHOLD:
                    print("!", end = "", flush = True)
                    silent_chunks = 0
                    has_started = True
                    audio_frames.append(audio_chunk)
                else:
                    if has_started:
                        print(".", end = '', flush = True)
                        silent_chunks += 1
                        audio_frames.append(audio_chunk)

                        if silent_chunks > silence_limit_chunks:
                            print("\n Silence detected. Stop recording.")
                            break
                        else:
                            pass
                if not audio_frames:
                    # print("No audio recorded")
                    return None

        full_audio = np.concatenate(audio_frames)
        write(config.WAV_OUTPUT_PATH, config.SAMPLE_RATE, full_audio)
        return str(config.WAV_OUTPUT_PATH)

    def _Transcribe_sound(self, audio_path):
        if not audio_path:
            print("No audio file to transcribe.")
            return ""

        print("Transcribe audio...")

        result = self.whisper_model.transcribe(audio_path, fp16 = False)
        text = result['text'].strip()

        print(f"User said: {text}")
        return text
    
    def _Chat_With_Ollama(self, user_text, image_frame = None, yolo_take = None ):
        # 1. Send a request to the local Ollama model
        # We use "chat" method for conversation-style interaction 
        message_reading = {
            'role': 'user',
            'content': user_text
        }
        
        print("Thinking...")
        
        if image_frame is not None:
            # Format image 
            gray_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
            is_success, buffer= cv2.imencode(".webp", 
                                            image_frame, 
                                            [int(cv2.IMWRITE_WEBP_QUALITY), 90]
                                            )
            if is_success:
                #From Image array change to bytes item
                image_bytes = buffer.tobytes()

                # Add image parameter to Ollama python
                message_reading['images'] = [image_bytes]

                #Tell Ollama go to watch image 
                message_reading['content'] = f"User said: '{user_text}'. Please look at the provided image and answer"

                print("[Brain] Image attached to prompt.")
            
            else:
                print("[Error] Failed to encode image frame.")

            
        if yolo_take:
            message_payload['content'] += f"(Note: YOLO detector found: {yolo_take})"


        config.ASK_HISTORY.append({'role': 'user', 'content': user_text})

        stream_response = ollama.chat(
            model = config.OLLAMA_MODEL_NAME,
            messages = config.ASK_HISTORY,
            stream = True
        )
        # 2. The 'response' is a dictionary. We need to extract the actual text.
        # Structure: response -> 'message' -> 'content'
        full_ai_response = ""
        print(f"{'-'* 3} AI Reply Bellow {'-'* 3}")
        for chunk in stream_response:
            part = chunk['message']['content']
            print(part, end = '', flush=True)
            full_ai_response += part
        
        print()

        config.ASK_HISTORY.append({'role': 'assistant', 'content': full_ai_response})

        return full_ai_response
    
    def _Speak(self, text):
        print("Speaking...")
        self.engine.say(text)
        self.engine.runAndWait()