# Two-Mic Transcriber

**Offline two-speaker transcription for dual wireless microphones (RØDE Wireless Micro, DJI Mic, Hollyland and similar).**

Record a conversation with two wireless mics, one per person, and get a transcript that says **who said what**, with timestamps:

```
[00:00:00] Alice: Hi, how are you?
[00:00:03] Bob: Good, thanks. Shall we start?
```

Everything runs **locally on your Mac**. No audio or text is uploaded anywhere, and there is no account or subscription.

## Why channels instead of voice recognition

Most transcription apps guess who is speaking by analysing voices ("speaker diarization"). That works most of the time, but can mix people up, especially with similar voices.

With two lapel mics in **Split mode**, you already know who is who: person 1 is on the left channel, person 2 on the right. This script uses that.

Each lapel mic also picks up the other person faintly. So instead of transcribing each channel separately (which would duplicate sentences), the script:

1. Transcribes the whole conversation once, with a timestamp for every word, using [Whisper](https://github.com/openai/whisper) (OpenAI's free speech-to-text model) running on your Mac
2. For every word, checks **which mic was louder** at that moment, and that's the speaker
3. Groups the words into turns and saves a clean transcript

## Requirements

- A Mac with Apple Silicon (M1 or newer) and macOS 13.5 (Ventura) or later
- A stereo recording with one speaker per channel

The transcription engine ([MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)) is optimised for Apple Silicon and doesn't run on Windows, Linux or Intel Macs.

## Recording setup

Tested with a **RØDE Wireless Micro (USB-C)** on an iPhone, through a USB-C to Lightning OTG adapter:

1. In **RØDE Central** (RØDE's free settings app), set the mics to **Split mode**. The default, Merged mode, mixes both people together and makes attribution impossible.
2. Record with **RØDE Reporter** (RØDE's free audio-only recording app) in **stereo**, not mono.
3. **Quick check:** play a test recording with headphones while only one person speaks. Their voice should be clearly louder in one ear.

Other dual-mic systems should work the same way, as long as the recording is stereo with one person per channel.

## Installation (one time, about 10 minutes)

1. **Install Python** from [python.org/downloads](https://www.python.org/downloads/). Run the installer like any other app.
2. **Install the tools.** Open **Terminal** (Cmd + Space, type "Terminal"), paste this and press Enter:
   ```
   python3 -m pip install -r requirements.txt
   ```
   (Run it from inside this folder. Or install directly: `python3 -m pip install mlx-whisper imageio-ffmpeg numpy`)
3. Download this repository (green **Code** button → **Download ZIP**) and unzip it, for example into `Documents`.

## Usage

1. Copy your recording to your Mac (AirDrop works well from RØDE Reporter).
2. In Terminal:
   ```
   cd ~/Documents/two-mic-transcriber
   python3 transcribe_conversation.py "recording.m4a" --left "Alice" --right "Bob" --language en
   ```
3. The transcript is saved next to the recording as `recording_transcript.txt`.

### Options

| Option | What it does |
|---|---|
| `--left "Name"` | Name of the person on the left channel (default: Speaker 1) |
| `--right "Name"` | Name of the person on the right channel (default: Speaker 2) |
| `--language xx` | Language code, e.g. `en`, `pt`, `es`, `fr`. Leave it out to auto-detect |
| `--margin 3` | How many dB louder one mic must be to switch speaker. Raise it if speakers switch too often, lower it if short replies get missed |
| `--model ...` | A different MLX Whisper model (default: `mlx-community/whisper-large-v3-turbo`) |

Supported formats: anything FFmpeg can read (M4A, WAV, MP3, AAC…).

## Good to know

- **First run** downloads the Whisper model (about 1.6 GB) once. After that it works fully offline.
- **Speed:** on an M1 MacBook Air, a 40-minute conversation takes roughly 5–15 minutes.
- **Accuracy:** very short interjections ("ok", "yes") said over the other person may occasionally be attributed to the wrong speaker.
- If both channels contain the same audio (mics not in Split mode, or recorded in mono), the script stops and tells you.

## Privacy

Audio and transcripts never leave your computer. The only internet connection is the one-time model download from Hugging Face.

**If you fork this repo, don't commit your recordings or transcripts.** The included `.gitignore` excludes common audio files and `*_transcript.txt` by default.

## License

[MIT](LICENSE): free to use, modify and share.
