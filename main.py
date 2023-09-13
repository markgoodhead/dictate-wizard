import os
os.environ["KIVY_LOG_MODE"] = "PYTHON"
if cores := os.cpu_count():
    os.environ["OMP_NUM_THREADS"] = str(cores)
import wavio
import io
import sys
import json
import requests
from io import StringIO
from functools import partial
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
Config.set('graphics', 'width', '600')
Config.set('graphics', 'height', '400')
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.properties import StringProperty, DictProperty, ObjectProperty
from kivy.utils import get_color_from_hex
from kivy.uix.modalview import ModalView
from kivy.uix.checkbox import CheckBox
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.popup import Popup
from kivy.clock import Clock

# Needed for Conjecture API
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

whisper_api_url = 'https://api.openai.com/v1/audio/transcriptions'
conjecture_api_url = 'https://api.conjecture.dev/v1/transcriptions'

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
    LOCAL_WHISPER = "Local Whisper"
    OPENAI = "OpenAI"
    CONJECTURE = "Conjecture"
    SONIOX = "Soniox"
    
@dataclass
class ProviderConfig:
    main_provider: Provider
    activated_providers: Set[Provider]
    
    def __post_init__(self):
        if self.main_provider not in self.activated_providers:
            raise ValueError(f"Main provider {self.main_provider.value} is not in the set of activated providers")
    
provider_config = ProviderConfig(main_provider=Provider.LOCAL_WHISPER, activated_providers={
    Provider.LOCAL_WHISPER})

batch_providers = {Provider.CONJECTURE, Provider.LOCAL_WHISPER, Provider.OPENAI}
streaming_providers = {Provider.SONIOX}

class ModelSize(Enum):
    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    LARGE = "large-v2"

def make_whisper_model(model_size: ModelSize = ModelSize.BASE_EN):
    return WhisperModel(model_size.value, device="cpu", compute_type="auto")

model = make_whisper_model()
    
provider_keys = {provider.name: "" for provider in Provider}
class WrappedLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            width=lambda *x:
            self.setter('text_size')(self, (self.width, None)),
            texture_size=lambda *x: self.setter('height')(self, self.texture_size[1]))

def create_property(name):
    return StringProperty()

def create_label(text, font_size='15sp', underlined=False, size_hint_x=1.0):
    if underlined:
        text = f"[u]{text}[/u]"
    return Label(text=text, font_size=font_size, size_hint_x=size_hint_x, color=get_color_from_hex("#D1D1D1"), markup=True)  # Light grey text

def create_wrapped_label(text, font_size='15sp', size_hint_x=1.0):
    return WrappedLabel(text=text, font_size=font_size, color=get_color_from_hex("#D1D1D1"), size_hint_x=size_hint_x)

def create_input(text, on_validate, font_size='15sp', size_hint_x=1.0):
    input_field = TextInput(
        text=text, multiline=False, size_hint_x=size_hint_x,
        size_hint_y=None, height=60, halign="center",
        background_color=get_color_from_hex("#5B5B5B"),  # Darker grey input background
        foreground_color=get_color_from_hex("#D1D1D1"),  # Light grey input text
        cursor_color=get_color_from_hex("#C67DD4"),  # Purple cursor
        font_size=font_size)
    input_field.bind(on_text_validate=on_validate)
    return input_field

def create_button(text, font_size='15sp', height=60):
    button = Button(
        text=text,
        size_hint_y=None,
        height=height,
        color=get_color_from_hex("#D1D1D1"),
        font_size=font_size
    )
    return button

class LoadingDialog(Popup):
    def __init__(self, **kwargs):
        super(LoadingDialog, self).__init__(**kwargs)
        self.title = "Loading model..."
        self.auto_dismiss = False
        self.log_message = Label()
        self.content = self.log_message
        self.is_loading_complete = False

    def set_message(self, message):
        def update_message(dt):
            self.log_message.text = message
        Clock.schedule_once(update_message)

    def on_open(self):
        self.log_message.text = "Loading..."

    def on_dismiss(self):
        self.log_message.text = ""
        
class ErrorPopup(Popup):
    def __init__(self, error_message, **kwargs):
        super(ErrorPopup, self).__init__(**kwargs)
        self.title = "Error"
        self.size_hint = (0.8, 0.4)
        
        content_layout = BoxLayout(orientation='vertical', padding=10)
        error_label = Label(text=error_message)
        dismiss_button = Button(text="Dismiss", size_hint=(1, 0.3))
        dismiss_button.bind(on_release=self.dismiss)
        
        content_layout.add_widget(error_label)
        content_layout.add_widget(dismiss_button)
        
        self.content = content_layout
        

class RecorderGUI(BoxLayout):
    error_popup = ObjectProperty(None)
    hotkey = StringProperty("x")
    modifiers = StringProperty("ctrl+alt")
    recording_status = StringProperty("Not Recording")
    processing_status = StringProperty("Not Processing")
    provider_keys = DictProperty({provider.name: "" for provider in Provider})
    provider_last_transcription = DictProperty({provider.name: "" for provider in Provider})
    provider_processing_time = DictProperty({provider.name: "" for provider in Provider})
    for provider in Provider:
        locals()[f"{provider.name.lower()}_key"] = create_property(provider.name)

    def __init__(self, **kwargs):
        super(RecorderGUI, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 20
        self.padding = 20
        
        self.provider_buttons = []
        
        self.load_api_keys()
        self.load_hotkey_config()

        with self.canvas.before:
            Color(*get_color_from_hex("#0C1441"))
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

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
        
        self.provider_modal_view = ModalView(size_hint=(0.8, 0.8))
        provider_layout = BoxLayout(orientation='vertical')
        for provider in Provider:
            box = BoxLayout(orientation='horizontal')
            box.add_widget(Label(text=provider.value))
            checkbox = CheckBox(active=provider in provider_config.activated_providers)
            checkbox.bind(active=partial(self.on_provider_check, provider))
            box.add_widget(checkbox)
            provider_layout.add_widget(box)
        self.provider_modal_view.add_widget(provider_layout)
        self.provider_modal_button = create_button(text='Select Active Providers')
        self.provider_modal_button.bind(on_release=self.provider_modal_view.open)
        
        self.provider_dropdown = DropDown()
        self.provider_dropdown.bind(on_select=self.on_provider_dropdown_select)
        self.provider_dropdown_button = create_button(text=f'Selected Main Provider: {provider_config.main_provider.value}')
        self.provider_dropdown_button.bind(on_release=self.provider_dropdown.open)
        
        provider_buttons_layout = BoxLayout(orientation='horizontal', height=60)
        provider_buttons_layout.add_widget(self.provider_modal_button)
        provider_buttons_layout.add_widget(self.provider_dropdown_button)
        self.add_widget(provider_buttons_layout)
        
        titles_layout = BoxLayout(orientation='horizontal', height=60)
        titles_layout.add_widget(create_label(text="Providers", underlined=True, size_hint_x=0.5))
        titles_layout.add_widget(create_label(text="Transcription", underlined=True))
        titles_layout.add_widget(create_label(text="Timing", underlined=True, size_hint_x=0.3))
        titles_layout.add_widget(create_label(text="API Key Input", underlined=True, size_hint_x=0.5))
        titles_layout.add_widget(create_label(text="API Key", underlined=True, size_hint_x=0.5))
        self.add_widget(titles_layout)
        
        for provider in provider_config.activated_providers:
            self.on_provider_check(provider, None, provider in provider_config.activated_providers)

        self.bind(recording_status=self.recording_status_label.setter('text'))
        self.bind(processing_status=self.processing_status_label.setter('text'))
        self.bind(hotkey=self.hotkey_value.setter('text'))
        self.bind(modifiers=self.modifiers_value.setter('text'))
        
    def on_provider_dropdown_select(self, instance, x):
        selected_provider = Provider[x.upper().replace(" ", "_")]
        provider_config.main_provider = selected_provider
        self.provider_dropdown_button.text = f'Selected Main Provider: {selected_provider.value}'
        
    def on_provider_check(self, provider, instance, value):
        if value:
            provider_config.activated_providers.add(provider)
            self.create_provider_ui(provider)
            btn = create_button(text=provider.value, height=50)
            btn.bind(on_release=lambda btn: self.provider_dropdown.select(btn.text))
            self.provider_dropdown.add_widget(btn)
            self.provider_buttons.append(btn)
        else:
            provider_config.activated_providers.discard(provider)
            self.clear_provider_ui(provider)
            for btn in self.provider_buttons:
                if btn.text == provider.value:
                    self.provider_dropdown.remove_widget(btn)
                    self.provider_buttons.remove(btn)
                    break
            if provider == provider_config.main_provider:
                provider_config.main_provider = next(iter(provider_config.activated_providers), None)
                self.provider_dropdown_button.text = f'Selected Main Provider: {provider_config.main_provider.value}'

    def update_provider_key_value(self, provider):
        def update_text(*args):
            setattr(self, f"{provider.name.lower()}_key", self.provider_keys[provider.name])
            getattr(self, f"{provider.name.lower()}_key_value").text = self.provider_keys[provider.name]
        return update_text
    
    def update_provider_transcription_value(self, provider):
        def update_text(*args):
            getattr(self, f"{provider.name.lower()}_last_transcription").text = self.provider_last_transcription[provider.name]
        return update_text

    def update_provider_processing_time_value(self, provider):
        def update_text(*args):
            getattr(self, f"{provider.name.lower()}_processing_time").text = self.provider_processing_time[provider.name]
        return update_text
    
    def create_provider_ui(self, provider):
        setattr(self, f"{provider.name.lower()}_key", self.provider_keys[provider.name])

        key_layout = BoxLayout(orientation='horizontal', height=60)
        key_layout.add_widget(create_label(text=f"{provider.value}", size_hint_x=0.5))
        
        last_transcription_label = create_wrapped_label(text=self.provider_last_transcription[provider.name], font_size='10sp')
        setattr(self, f"{provider.name.lower()}_last_transcription", last_transcription_label)
        key_layout.add_widget(last_transcription_label)

        processing_time_label = create_label(text=self.provider_processing_time[provider.name], size_hint_x=0.3)
        setattr(self, f"{provider.name.lower()}_processing_time", processing_time_label)
        key_layout.add_widget(processing_time_label)
        if provider == Provider.LOCAL_WHISPER:
            self.model_dropdown = DropDown()
            self.model_dropdown.bind(on_select=self.on_model_change)
            self.model_dropdown_button = create_button(text=f'Selected Model: {ModelSize.BASE_EN.value}')
            self.model_dropdown_button.bind(on_release=self.model_dropdown.open)
            for model_size in ModelSize:
                btn = create_button(text=model_size.value, height=50)
                btn.bind(on_release=lambda btn: self.model_dropdown.select(btn.text))
                self.model_dropdown.add_widget(btn)
            key_layout.add_widget(self.model_dropdown_button)
        else:
            key_input = create_input(text=self.provider_keys[provider.name], 
                                    on_validate=partial(self.on_key_validate, provider), size_hint_x=0.5)
            setattr(self, f"{provider.name.lower()}_key_input", key_input)
            key_layout.add_widget(key_input)
            key_value = create_wrapped_label(text=self.provider_keys[provider.name], font_size='10sp', size_hint_x=0.5)
            setattr(self, f"{provider.name.lower()}_key_value", key_value)
            key_layout.add_widget(key_value)
            self.bind(**{f"{provider.name.lower()}_key": key_value.setter('text')})
            self.bind(provider_keys=self.update_provider_key_value(provider))

        self.add_widget(key_layout)
        setattr(self, f"{provider.name.lower()}_key_layout", key_layout)
        self.bind(provider_last_transcription=self.update_provider_transcription_value(provider))
        self.bind(provider_processing_time=self.update_provider_processing_time_value(provider))
        
    def on_model_change(self, instance, value):
        loading_dialog = LoadingDialog()
        loading_dialog.open()
        
        class CustomStream(StringIO):
            def write(self, message):
                last_line = message.split('\n')[-1]
                loading_dialog.set_message(last_line)

            def flush(self):
                pass

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        custom_stream = CustomStream()
        sys.stdout = custom_stream
        sys.stderr = custom_stream

        def load_model():
            global model
            try:
                model = make_whisper_model(ModelSize(value))
                loading_dialog.is_loading_complete = True
                Clock.schedule_once(lambda dt: loading_dialog.dismiss())
                self.model_dropdown_button.text = f'Selected Model: {value}'
            except Exception as e:
                Clock.schedule_once(lambda dt: loading_dialog.dismiss())
                def set_error_popup(e):
                    self.error_popup = ErrorPopup(str(e))
                    self.error_popup.open()
                Clock.schedule_once(lambda dt, e=e: set_error_popup(e))
            
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        Thread(target=load_model).start()
        
    def clear_provider_ui(self, provider):
        key_layout = getattr(self, f"{provider.name.lower()}_key_layout")
        self.remove_widget(key_layout)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
        
    def load_api_keys(self):
        global provider_keys
        try:
            with open("api_keys.json", "r") as f:
                config = json.load(f)
            for provider in Provider:
                self.provider_keys[provider.name] = config[provider.name.lower() + "_key"]
                provider_keys[provider.name] = config[provider.name.lower() + "_key"]
            set_api_key(provider_keys["SONIOX"])
        except (FileNotFoundError, KeyError):
            print("API keys not found, please enter them manually")
        
    def save_api_keys(self, provider_keys):
        config = {provider.lower() + "_key": key for provider, key in provider_keys.items()}
        with open("api_keys.json", "w") as f:
            json.dump(config, f)
            
    def on_key_validate(self, provider, instance):
        global provider_keys

        provider_keys[provider.name] = instance.text
        self.provider_keys[provider.name] = instance.text

        if provider == Provider.SONIOX:
            set_api_key(instance.text)

        self.save_api_keys(provider_keys)

    def on_hotkey_validate(self, instance):
        self.update_hotkey_and_modifiers(instance.text, self.modifiers_input.text)
        self.save_hotkey_config(instance.text, self.modifiers_input.text)

    def on_modifiers_validate(self, instance):
        self.update_hotkey_and_modifiers(self.hotkey_input.text, instance.text)
        self.save_hotkey_config(self.hotkey_input.text, instance.text)

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
            
    def load_hotkey_config(self):
        global HOTKEY, MODIFIERS
        try:
            with open("hotkey_config.json", "r") as f:
                config = json.load(f)
            HOTKEY = KeyCode.from_char(config["hotkey"].lower())
            self.hotkey = config["hotkey"]
            if config["modifiers"]:
                MODIFIERS = {Key[modifier.lower()] for modifier in config["modifiers"]}
            else:
                MODIFIERS = set()
            self.modifiers = "+".join(config["modifiers"])
            print(f"Loaded hotkey: {config['hotkey']} and modifiers: {self.modifiers}")
        except (FileNotFoundError, KeyError):
            print("Hotkey configuration not found, please enter them manually")

    def save_hotkey_config(self, hotkey, modifiers):
        config = {"hotkey": hotkey, "modifiers": modifiers.split('+')}
        with open("hotkey_config.json", "w") as f:
            json.dump(config, f)


class RecorderApp(App):
    title = 'Dictate Wizard'
    icon = 'dictate_wizard.png'
    def build(self):
        return RecorderGUI()
    
audio_queue = Queue()

def record_callback(indata, frames, time, status):
    global audio, audio_queue
    audio_queue.put(indata.copy())
    audio.append(indata.copy())
    
def iter_audio_queue() -> Iterable[bytes]:
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
        app = App.get_running_app()
        app.root.provider_last_transcription[Provider.SONIOX.name] = f"{transcript}"
        app.root.provider_processing_time[Provider.SONIOX.name] = f"{time.time() - start_time:.2f} s"
        if Provider.SONIOX == provider_config.main_provider:
            print_transcript(transcript)
            app.root.processing_status = "Not Processing"
        print(f"Transcription from {Provider.SONIOX.value}: {transcript}")
        print(f"Processing {Provider.SONIOX.value} transcription request took {time.time() - start_time:.2f} seconds")

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
    app = App.get_running_app()
    app.root.provider_last_transcription[provider.name] = f"{transcript}"
    app.root.provider_processing_time[provider.name] = f"{time.time() - start_time:.2f} s"
    if provider == provider_config.main_provider and transcript:
        print_transcript(transcript)
    print(f"Transcription from {provider.value}: {transcript}")
    print(f"Processing {provider.value} transcription request took {time.time() - start_time:.2f} seconds")
    return transcript
    
def local_whisper_transcribe(audio_buffer):
    global model
    segments, _ = model.transcribe(audio_buffer)
    transcript = ""
    for segment in segments:
        transcript += segment.text
    return transcript.lstrip()
    
def openai_transcribe(audio_buffer):
    files = {"file": ("audio.wav", audio_buffer, "audio/wav")}
    data = {"model": "whisper-1"}
    headers = {"Authorization": f"Bearer {provider_keys[Provider.OPENAI.name]}"}

    response = requests.post(whisper_api_url, headers=headers, data=data, files=files)
    #print(response.json())
    if response.status_code == 200:
        transcript = response.json()["text"]
        return transcript
    else:
        print(f'Error: {response.status_code} - {response.text}')
        return None
    
def conjecture_transcribe(audio_buffer):
    headers = {"api-key": f"{provider_keys[Provider.CONJECTURE.name]}", "language": "en", "diarisation": "false", "word_timestamps": "false"}
    files = {"file": audio_buffer}
    response = requests.post(conjecture_api_url, headers=headers, files=files, verify=False)
    #print(response.json())
    if response.status_code == 201:
        transcript = response.json()["transcription"]["text"]
        return transcript
    else:
        print(f'Error: {response.status_code} - {response.text}')
        return None

def start_listener():
    def for_canonical(f):
        return lambda k: f(listener.canonical(k))
    with Listener(on_press=for_canonical(on_press), on_release=for_canonical(on_release)) as listener:
        listener.join()

if __name__ == '__main__':
    listener_thread = Thread(target=start_listener, daemon=True)
    listener_thread.start()
    print("Press your hotkey combination to start recording.")
    RecorderApp().run()
