api_key = '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'
conjecture_key = 'sk-m42iTB1N0w0kNM0QzeOA'
import os
if cores := os.cpu_count():
    os.environ["OMP_NUM_THREADS"] = str(cores)
import wavio
import io
import requests
import numpy as np
from pynput.keyboard import Key, KeyCode, Listener, Controller
import sounddevice as sd
import platform
import pyperclip
import time
from faster_whisper import WhisperModel
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


whisper_api_url = 'https://api.openai.com/v1/audio/transcriptions'
conjecture_api_url = 'https://api.conjecture.dev/transcribe'

HOTKEY = KeyCode.from_char('x')
MODIFIERS = {Key.alt, Key.ctrl}
keyboard = Controller()
current_pressed_modifiers = set()
is_recording = False
use_api = False
use_openai = False
if not use_api:
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
audio = []

def record_callback(indata, frames, time, status):
    audio.append(indata.copy())

def on_press(key):
    global current_pressed_modifiers, audio, is_recording, stream
    
    if key in MODIFIERS:
        current_pressed_modifiers.add(key)

    if all(modifier in current_pressed_modifiers for modifier in MODIFIERS) and key == HOTKEY and not is_recording:
        is_recording = True
        print("Hotkey pressed. Start recording.")
        audio = []
        stream = sd.InputStream(samplerate=16000, channels=1, callback=record_callback)
        stream.start()

def on_release(key):
    global current_pressed_modifiers, audio, is_recording, stream

    if key in current_pressed_modifiers:
        current_pressed_modifiers.remove(key)

    if key == HOTKEY and is_recording:
        is_recording = False
        print("Hotkey released. Stop recording.")
        start_time = time.time()
        stream.stop()
        stream.close()
        if audio:
            audio_data = np.concatenate(audio, axis=0)
            transcript = transcribe_audio(audio_data, start_time)
            if transcript:
                print("Transcription:", transcript)
                type_time = time.time()
                pyperclip.copy(transcript)
                if platform.system() == "Darwin":
                    keyboard.press(Key.cmd)
                    keyboard.press('v')
                    keyboard.release('v')
                    keyboard.release(Key.cmd)
                else:
                    keyboard.press(Key.ctrl)
                    keyboard.press('v')
                    keyboard.release('v')
                    keyboard.release(Key.ctrl)
                print(f"Typing transcript took {time.time() - type_time:.2f} seconds")
            else:
                print("No transcription returned.")
        else:
            print("No audio recorded.")


def transcribe_audio(audio_data, start_time):
    audio_buffer = io.BytesIO()
    wavio.write(audio_buffer, audio_data, 16000, sampwidth=2)
    #wavio.write("test_sound.wav", audio_data, 16000, sampwidth=2)
    audio_buffer.seek(0)
    http_time = time.time()
    print(f"Processing audio file took {http_time - start_time:.2f} seconds")
    if use_api:
        if use_openai:
            files = {"file": ("audio.wav", audio_buffer, "audio/wav")}
            data = {"model": "whisper-1"}
            headers = {"Authorization": f"Bearer {api_key}"}

            response = requests.post(whisper_api_url, headers=headers, data=data, files=files)
            #print(response.json())
            if response.status_code == 200:
                transcript = response.json()["text"]
                print(f"Processing HTTP request took {time.time() - http_time:.2f} seconds")
                return transcript
            else:
                print(f'Error: {response.status_code} - {response.text}')
                return None
        else:
            headers = {"Authorization": f"Bearer {conjecture_key}"}#, "Content-Type": "multipart/form-data"}
            files = {"file": audio_buffer, "language": "en", "diarize": False, "word_timestamps": False}
            response = requests.post(conjecture_api_url, headers=headers, files=files, verify=False)
            #print(response.json())
            if response.status_code == 201:
                transcript = response.json()["data"]["text"]
                print(f"Processing HTTP request took {time.time() - http_time:.2f} seconds")
                return transcript
            else:
                print(f'Error: {response.status_code} - {response.text}')
                return None
    else:
        segments, _ = model.transcribe(audio_buffer)
        transcript = ""
        for segment in segments:
            transcript += segment.text
        print(f"Processing local transcript took {time.time() - http_time:.2f} seconds")
        return transcript


if __name__ == '__main__':
    print("Press ctrl + alt + x to start recording.")
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
