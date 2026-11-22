# ComfyUI Inworld Voice Isolator

ComfyUI custom node integration for **Inworld Voice Isolator**, providing 48 kHz studio-grade vocal separation and background extraction.

---

## Key Features

- **Dual Engine Support**:
  - `offline (high quality)`: 64M parameter offline model using crossfaded chunks for maximum fidelity.
  - `streaming (realtime)`: 58M parameter causal model with low lookahead for fast processing.
- **Automatic or Manual Model Loading**:
  - Automatically downloads model weights from Hugging Face on first run if internet is available.
  - Supports offline / manual downloads placed into `ComfyUI/models/inworld/`.
- **VRAM-Optimized Long Audio Processing**:
  - Memory-efficient host accumulation keeps peak VRAM usage flat (< 2.4 GB) regardless of audio duration.
  - Seamlessly handles recordings spanning from seconds to over an hour.
  - Tuned selective-scan block recurrence for fast GPU inference.
- **Stereo & Mono Flexibility**:
  - `mono`: Downmixes to mono for efficient processing.
  - `stereo (dual-channel)`: Separates left and right channels independently to preserve stereo imaging.
  - `centered stereo`: Downmixes to mono and balances across both channels.
- **Background Extraction**:
  - Automatically isolates the background track (music, reverb, ambient noise) via phase-aligned input subtraction.
- **User Experience**:
  - Full integration with ComfyUI's live progress bar.
  - Native UI cancellation support.

---

## Installation

### Method 1: ComfyUI Manager
Install via **ComfyUI Manager** using the Git URL:
```text
https://github.com/Oyaabuun/ComfyUI-Inworld-Voice-Isolator
```

### Method 2: Manual Clone
Open a terminal in your `ComfyUI/custom_nodes/` folder:
```bash
cd custom_nodes
git clone https://github.com/Oyaabuun/ComfyUI-Inworld-Voice-Isolator.git
cd ComfyUI-Inworld-Voice-Isolator
pip install -r requirements.txt
```

---

## Model Weights

### 1. Automatic Download (Default)
Leave `model_name` set to `auto (download from HF if missing)`. The weights will automatically download to your Hugging Face cache on the first run.

### 2. Manual / Offline Download
If your machine is offline or behind a firewall, download the model weights directly and place them in `ComfyUI/models/inworld/`:

- **Offline HQ Model (64M)**: [denoiser.safetensors](https://huggingface.co/inworld/voice-isolator/blob/main/denoiser.safetensors) (~256 MB)
- **Streaming Realtime Model (58M)**: [denoiser_realtime.safetensors](https://huggingface.co/inworld/voice-isolator/blob/main/denoiser_realtime.safetensors) (~232 MB)

Files in `ComfyUI/models/inworld/` will automatically appear in the `model_name` dropdown menu.

---

## Node Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | AUDIO | *Required* | Input audio from `LoadAudio` or any audio node. |
| `model_name` | Combo | `auto (download from HF if missing)` | Select `auto` to fetch from HF, or select any local checkpoint in `models/inworld/`. |
| `mode` | Combo | `offline (high quality)` | Model mode (`offline (high quality)` or `streaming (realtime)`). |
| `device` | Combo | `auto` | Execution device (`auto`, `cuda`, or `cpu`). |
| `channel_handling` | Combo | `mono` | Channel mode (`mono`, `stereo (dual-channel)`, `centered stereo`). |
| `chunk_seconds` | Float | `8.0` | Window size for offline mode (2.0s to 30.0s). |
| `overlap_seconds` | Float | `2.0` | Crossfade overlap duration (0.5s to 8.0s). |
| `output_sample_rate` | Combo | `original` | Target sample rate (`original` or `48000`). |

### Outputs
- `voice`: Clean isolated speech audio.
- `background`: Extracted residual background noise, music, or atmosphere.

---

## Example Workflow

An example workflow is available in `example_workflows/inworld_voice_isolator.json`. You can drag and drop this JSON file directly into ComfyUI.

---

## License

This project is licensed under the Apache License 2.0.
