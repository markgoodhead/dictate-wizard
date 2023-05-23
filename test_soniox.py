from typing import Iterable
from soniox.transcribe_live import transcribe_stream
from soniox.speech_service import SpeechClient, set_api_key

set_api_key("a5a8475630adcccc88eff3788828b8bc39feeb26ce1fefba904fa51d121806cb")


def iter_audio() -> Iterable[bytes]:
    with open("test_sound.wav", "rb") as fh:
        while True:
            audio = fh.read(1024)
            if len(audio) == 0:
                break
            print(type(audio), len(audio))
            yield audio


def main():
    with SpeechClient() as client:
        for result in transcribe_stream(iter_audio(), client):
            # Variable result contains final and non-final words.
            print(" ".join(w.text for w in result.words))


if __name__ == "__main__":
    main()