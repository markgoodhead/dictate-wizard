api_key = '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'
conjecture_key = 'sk-m42iTB1N0w0kNM0QzeOA'
import os
os.environ['WM_CLASS'] = "Dictate Wizard"
os.environ['SDL_VIDEO_X11_WMCLASS'] = "Dictate Wizard"
if cores := os.cpu_count():
    os.environ["OMP_NUM_THREADS"] = str(cores)
import wavio
import io
import requests
import numpy as np
from copy import deepcopy
from typing import Iterable
from enum import Enum
from dataclasses import dataclass
from typing import Set
from threading import Thread
import numpy as np
from pynput.keyboard import Key, KeyCode, Listener, Controller
import sounddevice as sd
import platform
import pyperclip
import time
from faster_whisper import WhisperModel
import urllib3
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from soniox.transcribe_live import transcribe_stream
from soniox.speech_service import SpeechClient, set_api_key
from kivy.config import Config
Config.set('graphics', 'width', '400')
Config.set('graphics', 'height', '300')
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.properties import StringProperty
from kivy.core.text import LabelBase
from kivy.utils import get_color_from_hex

LabelBase.register(name='Roboto',
                   fn_regular='Roboto-Regular.ttf')

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

whisper_api_url = 'https://api.openai.com/v1/audio/transcriptions'
conjecture_api_url = 'https://api.conjecture.dev/transcribe'
set_api_key("a5a8475630adcccc88eff3788828b8bc39feeb26ce1fefba904fa51d121806cb")

HOTKEY = KeyCode.from_char('x')
MODIFIERS = {Key.alt, Key.ctrl}
keyboard = Controller()
current_pressed_modifiers = set()
is_recording = False
write_recording = False
audio = []
model = None
start_time = None

class Provider(Enum):
    OPENAI = "OpenAI"
    LOCAL_WHISPER = "Local Whisper"
    CONJECTURE = "Conjecture"
    SONIOX = "Soniox"
    
@dataclass
class ProviderConfig:
    main_provider: Provider
    activated_providers: Set[Provider]
    
    def __post_init__(self):
        if self.main_provider not in self.activated_providers:
            raise ValueError(f"Main provider {self.main_provider} is not in the set of activated providers")
    
provider_config = ProviderConfig(main_provider=Provider.SONIOX, activated_providers={
    Provider.OPENAI, 
    Provider.CONJECTURE, 
    Provider.SONIOX, 
    Provider.LOCAL_WHISPER})

batch_providers = {Provider.CONJECTURE, Provider.LOCAL_WHISPER, Provider.OPENAI}
streaming_providers = {Provider.SONIOX}

if Provider.LOCAL_WHISPER in provider_config.activated_providers:
    model = WhisperModel("base.en", device="cpu", compute_type="int8")

class WrappedLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            width=lambda *x:
            self.setter('text_size')(self, (self.width, None)),
            texture_size=lambda *x: self.setter('height')(self, self.texture_size[1]))

class RecorderGUI(BoxLayout):
    last_translation = StringProperty("")
    processing_time = StringProperty("")
    hotkey = StringProperty("x")
    modifiers = StringProperty("ctrl+alt")
    recording_status = StringProperty("Not Recording")
    processing_status = StringProperty("Not Processing")

    def __init__(self, **kwargs):
        super(RecorderGUI, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 10
        self.padding = 10

        with self.canvas.before:
            Color(*get_color_from_hex("#0C1441"))
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        def create_label(text):
            return Label(text=text, font_size='16sp', color=get_color_from_hex("#D1D1D1"))  # Light grey text
        
        def create_input(text, on_validate):
            input_field = TextInput(
                text=text, multiline=False, 
                size_hint_y=None, height=60, halign="center",
                background_color=get_color_from_hex("#5B5B5B"),  # Darker grey input background
                foreground_color=get_color_from_hex("#D1D1D1"),  # Light grey input text
                cursor_color=get_color_from_hex("#C67DD4"))  # Purple cursor
            input_field.bind(on_text_validate=on_validate)
            return input_field

        last_translation_layout = BoxLayout(orientation='horizontal', height=60)
        last_translation_layout.add_widget(create_label(text="Last transcription:"))
        self.last_translation_label = WrappedLabel(text=self.last_translation, font_name='Roboto', font_size='12sp', color=get_color_from_hex("#D1D1D1"))
        last_translation_layout.add_widget(self.last_translation_label)
        self.add_widget(last_translation_layout)

        processing_time_layout = BoxLayout(orientation='horizontal', height=60)
        processing_time_layout.add_widget(create_label(text="Processing Time:"))
        self.processing_time_label = create_label(text=self.processing_time)
        processing_time_layout.add_widget(self.processing_time_label)
        self.add_widget(processing_time_layout)

        recording_status_layout = BoxLayout(orientation='horizontal', height=60)
        recording_status_layout.add_widget(create_label(text="Recording status:"))
        self.recording_status_label = create_label(text=self.recording_status)
        recording_status_layout.add_widget(self.recording_status_label)
        self.add_widget(recording_status_layout)

        processing_status_layout = BoxLayout(orientation='horizontal', height=60)
        processing_status_layout.add_widget(create_label(text="Processing status:"))
        self.processing_status_label = create_label(text=self.processing_status)
        processing_status_layout.add_widget(self.processing_status_label)
        self.add_widget(processing_status_layout)
        
        hotkey_layout = BoxLayout(orientation='horizontal', height=60)
        hotkey_layout.add_widget(create_label(text="Hotkey:"))
        self.hotkey_input = create_input(text=self.hotkey, on_validate=self.on_hotkey_validate)
        hotkey_layout.add_widget(self.hotkey_input)
        self.hotkey_value = create_label(text=self.hotkey)
        hotkey_layout.add_widget(self.hotkey_value)
        self.add_widget(hotkey_layout)

        modifiers_layout = BoxLayout(orientation='horizontal', height=60)
        modifiers_layout.add_widget(create_label(text="Modifiers:"))
        self.modifiers_input = create_input(text=self.modifiers, on_validate=self.on_modifiers_validate)
        modifiers_layout.add_widget(self.modifiers_input)
        self.modifiers_value = create_label(text=self.modifiers)
        modifiers_layout.add_widget(self.modifiers_value)
        self.add_widget(modifiers_layout)

        self.bind(recording_status=self.recording_status_label.setter('text'))
        self.bind(processing_status=self.processing_status_label.setter('text'))
        self.bind(last_translation=self.last_translation_label.setter('text'))
        self.bind(processing_time=self.processing_time_label.setter('text'))
        self.bind(hotkey=self.hotkey_value.setter('text'))
        self.bind(modifiers=self.modifiers_value.setter('text'))

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def on_hotkey_validate(self, instance):
        self.update_hotkey_and_modifiers(instance.text, self.modifiers_input.text)

    def on_modifiers_validate(self, instance):
        self.update_hotkey_and_modifiers(self.hotkey_input.text, instance.text)

    def update_hotkey_and_modifiers(self, new_hotkey, new_modifiers):
        global HOTKEY, MODIFIERS
        try:
            HOTKEY = KeyCode.from_char(new_hotkey.lower())
            self.hotkey = new_hotkey
            if new_modifiers:
                MODIFIERS = {Key[modifier.lower()] for modifier in new_modifiers.split('+')}
            else:
                MODIFIERS = set()
            self.modifiers = new_modifiers
            print(f"Updating hotkey to {new_hotkey} and modifiers to {new_modifiers}")
        except:
            print(f"Invalid hotkey {new_hotkey} or modifiers {new_modifiers}.")

class RecorderApp(App):
    title = 'Dictate Wizard'
    icon = 'dictate_wizard.ico'
    def build(self):
        return RecorderGUI()
    
audio_queue = Queue()

def record_callback(indata, frames, time, status):
    global audio, audio_queue
    audio_queue.put(indata.copy())
    audio.append(indata.copy())
    
def iter_audio_queue() -> Iterable[bytes]:
    # This function yields audio data from the queue
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        audio = np.ascontiguousarray(audio.astype(np.int16), "<h")
        audio = audio.tobytes()
        assert isinstance(audio, bytes)
        yield audio
        
executor = ThreadPoolExecutor(max_workers=1)

def transcribe_audio_stream():
    global start_time
    # This function is run in a separate thread and continuously processes audio data
    with SpeechClient() as client:
        transcript = ""
        for result in transcribe_stream(iter_audio_queue(), 
                                        client, 
                                        audio_format="pcm_s16le", 
                                        sample_rate_hertz=16000, 
                                        num_audio_channels=1):
            for word in result.words:
                if word.is_final:
                    text = word.text
                    if transcript == "":
                        transcript += text
                    elif text in [".", "?", "!", ","]:
                        transcript += text
                    else:
                        transcript += " " + text
        print(f"Transcription from {Provider.SONIOX}: {transcript}")
        print(f"Processing {Provider.SONIOX} transcription request took {time.time() - start_time:.2f} seconds")
        if Provider.SONIOX == provider_config.main_provider:
            print_transcript(transcript)
            app = App.get_running_app()
            app.root.processing_status = "Not Processing"
            app.root.last_translation = f"{transcript}"
            app.root.processing_time = f"{time.time() - start_time:.2f} seconds"

def on_press(key):
    global current_pressed_modifiers, audio, is_recording, stream, HOTKEY, MODIFIERS, provider_config, batch_providers, streaming_providers
    
    if key in MODIFIERS:
        current_pressed_modifiers.add(key)

    if all(modifier in current_pressed_modifiers for modifier in MODIFIERS) and key == HOTKEY and not is_recording:
        is_recording = True
        app = App.get_running_app()
        app.root.recording_status = "Recording"
        print("Hotkey pressed. Start recording.")
        audio = []
        stream = sd.InputStream(samplerate=16000, channels=1,
                                callback=record_callback, dtype='int16', blocksize=1280)
        stream.start()
        if any(provider in provider_config.activated_providers for provider in streaming_providers):
            executor.submit(transcribe_audio_stream)

def on_release(key):
    global current_pressed_modifiers, audio, audio_queue, is_recording, stream, HOTKEY, provider_config, batch_providers, streaming_providers, start_time

    if key in current_pressed_modifiers:
        current_pressed_modifiers.remove(key)

    if key == HOTKEY and is_recording:
        is_recording = False
        app = App.get_running_app()
        app.root.recording_status = "Not Recording"
        app.root.processing_status = "Processing"
        print("Hotkey released. Stop recording.")
        start_time = time.time()
        stream.stop()
        stream.close()
        if any(provider in provider_config.activated_providers for provider in streaming_providers):
            audio_queue.put(None)
        if any(provider in provider_config.activated_providers for provider in batch_providers):
            if audio:
                audio_data = np.concatenate(audio, axis=0)
                audio_buffer = io.BytesIO()
                wavio.write(audio_buffer, audio_data, 16000, sampwidth=2)
                if write_recording:
                    wavio.write("last_recording.wav", audio_data, 16000, sampwidth=2)
                audio_buffer.seek(0)
                #print(f"Processing audio file took {time.time() - start_time:.2f} seconds")
                threads = []
                for provider in provider_config.activated_providers:
                    if provider in batch_providers:  
                        t = Thread(target=transcribe_audio_batch, args=(deepcopy(audio_buffer), provider))
                        t.start()
                        threads.append(t)
                for t in threads:
                    t.join()
            app.root.processing_status = "Not Processing"
        
def print_transcript(transcript):
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

def transcribe_audio_batch(audio_buffer, provider):
    global provider_config, start_time
    if provider == Provider.CONJECTURE:
        transcript = conjecture_transcribe(audio_buffer)
    elif provider == Provider.LOCAL_WHISPER:
        transcript = local_whisper_transcribe(audio_buffer)
    elif provider == Provider.OPENAI:
        transcript = openai_transcribe(audio_buffer)
    if provider == provider_config.main_provider and transcript:
        print_transcript(transcript)
        app = App.get_running_app()
        app.root.last_translation = f"{transcript}"
        app.root.processing_time = f"{time.time() - start_time:.2f} seconds"
    print(f"Transcription from {provider}: {transcript}")
    print(f"Processing {provider} transcription request took {time.time() - start_time:.2f} seconds")
    return transcript
    
def local_whisper_transcribe(audio_buffer):
    global model
    if not model:
        model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_buffer)
    transcript = ""
    for segment in segments:
        transcript += segment.text
    return transcript
    
def openai_transcribe(audio_buffer):
    files = {"file": ("audio.wav", audio_buffer, "audio/wav")}
    data = {"model": "whisper-1"}
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(whisper_api_url, headers=headers, data=data, files=files)
    #print(response.json())
    if response.status_code == 200:
        transcript = response.json()["text"]
        return transcript
    else:
        print(f'Error: {response.status_code} - {response.text}')
        return None
    
def conjecture_transcribe(audio_buffer):
    headers = {"Authorization": f"Bearer {conjecture_key}"}#, "Content-Type": "multipart/form-data"}
    files = {"file": audio_buffer, "language": "en", "diarize": False, "word_timestamps": False}
    response = requests.post(conjecture_api_url, headers=headers, files=files, verify=False)
    #print(response.json())
    if response.status_code == 201:
        transcript = response.json()["data"]["text"]
        return transcript
    else:
        print(f'Error: {response.status_code} - {response.text}')
        return None

def start_listener():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

if __name__ == '__main__':
    listener_thread = Thread(target=start_listener, daemon=True)
    listener_thread.start()
    print("Press ctrl + alt + x to start recording.")
    RecorderApp().run()
