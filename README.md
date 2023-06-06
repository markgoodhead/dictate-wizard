# 🪄 Dictate Wizard 🪄 

![Logo](dictate_wizard.ico)

![Discord Shield](https://discordapp.com/api/guilds/1115298962757402665/widget.png?style=shield)

Dictate Wizard is an open source dictation tool. The goal is to obsolete as much typing as possible and let you speak your emails, instant messages etc instead.

It supports local Whisper-based transcription (free, but lower accuracy) as well as multiple commercial providers like OpenAI, Soniox and Conjecture (higher accuracy but you need a paid API key). Users can select a single or multiple providers and compare the transcription results and processing time.

It features an interactive GUI with options to update API provider keys, toggle active providers, and designate a primary provider for the transcription (the one used to output the text to the keyboard). Dictate Wizard also lets users customize their hotkey and modifier keys to activate recording.

This project is written in Python and uses Kivy for the GUI. It's intended to be cross-platform (I've tested on MacOS and Windows but not Linux). It outputs via adding the transcription text into the clipboard and pasting it.

## Providers

Suggestions for alternative providers to be added are welcome (please open an Issue). Currently it supports:
- OpenAI https://platform.openai.com/docs/guides/speech-to-text/quickstart
- Conjecture https://platform.conjecture.dev/transcriptions
- Soniox https://soniox.com/products/speech-recognition-ai/

Soniox is the only provider supported in 'streaming' mode, i.e. the transcription happens concurrently with the audio recording. As such it's the fastest provider in the list to return an output as both the local Whisper and the other providers are all processed in a sequential fashion.

The local Whisper functionality is provided by https://github.com/guillaumekln/faster-whisper. It uses the base.en model as this is competitive in transcription time with the API providers. 

## Usage
1. Clone this repository:

```bash
git clone https://github.com/markgoodhead/dictate-wizard.git
```

2. Change to the project directory:

```bash
cd dictate-wizard
```

3. Install the prerequisites:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python main.py
```

5. Use the GUI to configure your API keys, select the providers you wish to use, and designate your hotkey and modifiers.

6. Activate recording by pressing and holding your selected hotkey combination (defaults to ctrl+alt+x). Speak into your microphone. Release your hotkey and the transcription will be output wherever your cursor is highlighted.

## Known Issues

Windows:
- The keyboard actions seem to take 300ms to process which adds extra delay to the text output
- Doesn't load the Wizard icon

## How to Contribute
Contributions are welcome! Please feel free to submit a pull request or open an issue. For major changes, please open an issue first to discuss what you would like to change.

## Roadmap

Wishlist for improvements:
- Support CoreML and GPUs for faster local Whisper inference
- Support streaming for local Whisper inference
- Enable selecting larger Whisper models locally (would require performance improvements to make practical)
- Test Windows and Linux versions
- Package Dictate Wizard up into a deployable, e.g. with PyInstaller

Feature requests welcomed too; please make an Issue or discuss in the Discord server.

## License
This project is licensed under the GNU General Public License v3.0 License. See the `LICENSE` file for more details.

## Acknowledgements
We are grateful to all the transcription providers whose services make this project possible.
