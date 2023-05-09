api_key = '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'
conjecture_key = 'sk-m42iTB1N0w0kNM0QzeOA'

import wavio
import io
import requests
import numpy as np
from pynput.keyboard import Key, KeyCode, Listener
import sounddevice as sd
import pyautogui
import platform
import pyperclip
import time
from whispercpp import Whisper

whisper_api_url = 'https://api.openai.com/v1/audio/transcriptions'
conjecture_api_url = 'https://api.conjecture.dev/transcribe'

HOTKEY = KeyCode.from_char('x')
pyautogui.FAILSAFE = False
alt_pressed = False
ctrl_pressed = False
is_recording = False
use_api = True
use_openai = False
if not use_api:
    w = Whisper.from_pretrained("base.en")
audio = []

def record_callback(indata, frames, time, status):
    audio.append(indata.copy())

def on_press(key):
    global alt_pressed, ctrl_pressed, audio, is_recording, stream

    if key == Key.alt:
        alt_pressed = True

    if key == Key.ctrl:
        ctrl_pressed = True

    if alt_pressed and ctrl_pressed and key == HOTKEY and not is_recording:
        is_recording = True
        print("Hotkey pressed. Start recording.")
        audio = []
        stream = sd.InputStream(samplerate=16000, channels=1, callback=record_callback)
        stream.start()

def on_release(key):
    global alt_pressed, ctrl_pressed, audio, is_recording, stream

    if key == Key.alt:
        alt_pressed = False

    if key == Key.ctrl:
        ctrl_pressed = False

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
                    pyautogui.hotkey("command", "shift", "v", interval=0.05)
                else:
                    pyautogui.hotkey("ctrl", "v")
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
            print(response.json())
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
            print(response.json())
            if response.status_code == 201:
                transcript = response.json()["data"]["text"]
                print(f"Processing HTTP request took {time.time() - http_time:.2f} seconds")
                return transcript
            else:
                print(f'Error: {response.status_code} - {response.text}')
                return None
    else:
        transcript = w.transcribe_from_file("test_sound.wav")
        print(f"Processing local transcript took {time.time() - http_time:.2f} seconds")
        return transcript


if __name__ == '__main__':
    print("Press ctrl + alt + x to start recording.")
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
