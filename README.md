# ComfyUI Inworld Voice Isolator

ComfyUI custom node integration for **Inworld Voice Isolator**, providing 48 kHz studio-grade vocal separation, background noise extraction, and multi-format audio export.

---

## Key Features

- **Dual Engine Support**:
  - `offline (high quality)`: 64M parameter offline model using crossfaded chunks for maximum fidelity.
  - `streaming (realtime)`: 58M parameter causal model with low lookahead for fast processing.
- **Multi-Format Audio Export (`Save Audio (Multi-Format)`)**:
  - Eliminates ComfyUI's deprecated `SaveAudio` node warning.
  - Direct export to **WAV** (uncompressed 16-bit PCM), **FLAC** (lossless compressed), or **MP3** (high-compatibility).
  - Selectable MP3 bitrate (`320k`, `256k`, `192k`, `128k`).
  - Interactive HTML5 audio player preview embedded in the node.
- **Automatic or Manual Model Loading**:
  - Automatically downloads model weights from Hugging Face on first run if internet is available.
  - Supports offline / manual downloads placed into `ComfyUI/models/diffusion_models/inworld/` (or `ComfyUI/models/diffusion_models/`).
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
  - Real-time progress bar in both the ComfyUI web interface and terminal console.
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
If your machine is offline or behind a firewall, download the model weights directly and place them in `ComfyUI/models/diffusion_models/inworld/` (or `ComfyUI/models/diffusion_models/`):

- **Offline HQ Model (64M)**: [denoiser.safetensors](https://huggingface.co/inworld/voice-isolator/blob/main/denoiser.safetensors) (~256 MB)
- **Streaming Realtime Model (58M)**: [denoiser_realtime.safetensors](https://huggingface.co/inworld/voice-isolator/blob/main/denoiser_realtime.safetensors) (~232 MB)

Files placed in `ComfyUI/models/diffusion_models/inworld/` will automatically appear in the `model_name` dropdown menu.

---

## Nodes Included

### 1. Inworld Voice Isolator
| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | AUDIO | *Required* | Input audio from `LoadAudio` or any audio node. |
| `mode` | Combo | `offline (high quality)` | Model mode (`offline (high quality)` or `streaming (realtime)`). |
| `device` | Combo | `auto` | Execution device (`auto`, `cuda`, or `cpu`). |
| `channel_handling` | Combo | `mono` | Channel mode (`mono`, `stereo (dual-channel)`, `centered stereo`). |
| `chunk_seconds` | Float | `8.0` | Window size for offline mode (2.0s to 30.0s). |
| `overlap_seconds` | Float | `2.0` | Crossfade overlap duration (0.5s to 8.0s). |
| `output_sample_rate` | Combo | `original` | Target sample rate (`original` or `48000`). |
| `model_name` | Combo | `auto (download from HF if missing)` | Select `auto` to fetch from HF, or select any local checkpoint in `diffusion_models/`. |

**Outputs**:
- `voice`: Clean isolated speech audio.
- `background`: Extracted residual background noise, music, or atmosphere.

### 2. Save Audio (Multi-Format)
| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | AUDIO | *Required* | Audio track to save. |
| `filename_prefix` | String | `inworld/isolated_voice` | Output file path and naming pattern. |
| `format` | Combo | `wav` | Target format: `wav`, `flac`, or `mp3`. |
| `mp3_quality` | Combo | `320k` | Bitrate for MP3: `320k`, `256k`, `192k`, `128k`. |

---

## Example Workflow

An example workflow is available in `example_workflows/inworld_voice_isolator.json`. You can drag and drop this JSON file directly into ComfyUI.

---

## License

This project is licensed under the Apache License 2.0.
