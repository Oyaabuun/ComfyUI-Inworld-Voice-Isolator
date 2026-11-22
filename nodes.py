import json
import math
import os
import sys
import torch
from inworld import voice_isolation
import inworld.voice_isolation.scan as scan
from inworld.voice_isolation.model import window_samples
import folder_paths
import comfy.audio
import comfy.model_management
import comfy.utils

# Optimize selective_scan frame block size to reduce peak VRAM from 7.2 GB to ~1.5-2.4 GB and accelerate scan
scan.BLOCK = 4

_CACHED_ISOLATORS = {}


class ConsoleProgressBar:
    """Synchronizes progress between the ComfyUI web frontend and terminal console."""
    def __init__(self, total: int):
        self.total = max(1, total)
        self.current = 0
        self.comfy_pbar = comfy.utils.ProgressBar(self.total)
        self._last_pct = -1

    def update(self, steps: int = 1):
        self.current += steps
        if self.current > self.total:
            self.current = self.total
        self.comfy_pbar.update(steps)
        pct = int((self.current / self.total) * 100)
        if pct != self._last_pct or self.current == self.total:
            self._last_pct = pct
            bar_len = 30
            filled = int(round(bar_len * self.current / self.total))
            bar = "=" * filled + "-" * (bar_len - filled)
            if self.current >= self.total:
                sys.stdout.write(f"\r[Inworld Voice Isolator] [{bar}] 100% ({self.total}/{self.total} chunks)\n")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\r[Inworld Voice Isolator] [{bar}] {pct}% ({self.current}/{self.total} chunks)")
                sys.stdout.flush()


def _get_inworld_model_list():
    found = []
    # Search in diffusion_models folder (e.g. models/diffusion_models/inworld/ or models/diffusion_models/)
    for f in folder_paths.get_filename_list("diffusion_models"):
        fl = f.lower()
        if "inworld" in fl or "denoiser" in fl:
            found.append(f)
    # Search in audio_encoders folder if present
    if "audio_encoders" in folder_paths.folder_names_and_paths:
        for f in folder_paths.get_filename_list("audio_encoders"):
            fl = f.lower()
            if "inworld" in fl or "denoiser" in fl:
                found.append(f)
    return ["auto (download from HF if missing)"] + sorted(list(set(found)))


def _resolve_model_and_mode(model_choice: str, mode_str: str):
    default_mode = "streaming" if "streaming" in mode_str else "offline"
    if model_choice == "auto (download from HF if missing)":
        return "inworld/voice-isolator", default_mode

    # Check diffusion_models first, then audio_encoders
    full_path = folder_paths.get_full_path("diffusion_models", model_choice)
    if not full_path and "audio_encoders" in folder_paths.folder_names_and_paths:
        full_path = folder_paths.get_full_path("audio_encoders", model_choice)

    if not full_path or not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file '{model_choice}' not found in models/diffusion_models/ or models/audio_encoders/!")

    # Inspect checkpoint metadata to auto-match streaming/offline mode if needed
    mode = default_mode
    if full_path.endswith(".safetensors"):
        try:
            import safetensors.torch
            with safetensors.safe_open(full_path, framework="pt") as f:
                meta = f.metadata() or {}
                if "config" in meta:
                    cfg = json.loads(meta["config"])
                    if cfg.get("causal") is True:
                        mode = "streaming"
                    elif cfg.get("causal") is False:
                        mode = "offline"
        except Exception:
            pass
    elif "realtime" in model_choice.lower() or "streaming" in model_choice.lower():
        mode = "streaming"

    return full_path, mode


def _get_isolator(model_target: str, mode: str, device_choice: str):
    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_choice

    key = (model_target, mode, device)
    if key not in _CACHED_ISOLATORS:
        _CACHED_ISOLATORS[key] = voice_isolation.VoiceIsolator(
            model_target,
            mode=mode,
            device=device,
        )
    return _CACHED_ISOLATORS[key]


def _isolate_mono_offline(
    isolator,
    audio_mono: torch.Tensor,
    sample_rate: int,
    target_sr: int,
    chunk_seconds: float,
    overlap_seconds: float,
    pbar: ConsoleProgressBar = None,
) -> torch.Tensor:
    if audio_mono.ndim == 1:
        audio_mono = audio_mono.unsqueeze(0)
    audio_mono = audio_mono.float().cpu()

    target_len = int(round(audio_mono.shape[-1] * target_sr / sample_rate))
    if sample_rate != 48000:
        audio_48k = comfy.audio.resample(audio_mono, sample_rate, 48000)
    else:
        audio_48k = audio_mono

    n = audio_48k.shape[-1]
    hop_length = isolator.model.config.hop_length
    chunk_samples, overlap_samples = window_samples(chunk_seconds, overlap_seconds, 48000, hop_length)

    if n <= chunk_samples:
        comfy.model_management.throw_exception_if_processing_interrupted()
        cin = audio_48k.to(isolator.device)
        with torch.inference_mode(), isolator._autocast():
            cout = isolator.model(cin).cpu()
        out_48k = cout
        if pbar:
            pbar.update(1)
    else:
        out_48k = torch.zeros(1, n, dtype=torch.float32)
        weight = torch.zeros(n, dtype=torch.float32)
        starts = list(range(0, n - chunk_samples, chunk_samples - overlap_samples)) + [n - chunk_samples]

        for index, start in enumerate(starts):
            comfy.model_management.throw_exception_if_processing_interrupted()
            end = start + chunk_samples
            w = torch.ones(chunk_samples, dtype=torch.float32)
            if index > 0:
                fade_in = starts[index - 1] + chunk_samples - overlap_samples - start
                w[:fade_in] = 0.0
                if overlap_samples > 0:
                    w[fade_in:fade_in + overlap_samples] = torch.linspace(0.0, 1.0, overlap_samples)
            if overlap_samples > 0 and end < n:
                w[-overlap_samples:] = torch.linspace(1.0, 0.0, overlap_samples)

            cin = audio_48k[:, start:end].to(isolator.device)
            with torch.inference_mode(), isolator._autocast():
                cout = isolator.model(cin).cpu()

            out_48k[:, start:end] += cout * w
            weight[start:end] += w

            if pbar:
                pbar.update(1)

            if (index + 1) % 50 == 0 and isolator.device.type == "cuda":
                comfy.model_management.soft_empty_cache()

        out_48k = (out_48k / weight.clamp(min=1e-6)).float()

    if target_sr != 48000:
        out_final = comfy.audio.resample(out_48k, 48000, target_sr).squeeze(0)
    else:
        out_final = out_48k.squeeze(0)

    if out_final.shape[-1] > target_len:
        out_final = out_final[:target_len]
    elif out_final.shape[-1] < target_len:
        out_final = torch.nn.functional.pad(out_final, (0, target_len - out_final.shape[-1]))

    return out_final


def _isolate_mono_streaming(
    isolator,
    audio_mono: torch.Tensor,
    sample_rate: int,
    target_sr: int,
    pbar: ConsoleProgressBar = None,
) -> torch.Tensor:
    if audio_mono.ndim == 1:
        audio_mono = audio_mono.unsqueeze(0)
    audio_mono = audio_mono.float().cpu()

    target_len = int(round(audio_mono.shape[-1] * target_sr / sample_rate))
    if sample_rate != 48000:
        audio_48k = comfy.audio.resample(audio_mono, sample_rate, 48000)
    else:
        audio_48k = audio_mono

    n = audio_48k.shape[-1]
    session = isolator.stream()
    piece_samples = int(0.5 * 48000)  # 24000 samples per half-second piece
    pieces = []

    for i in range(0, n, piece_samples):
        comfy.model_management.throw_exception_if_processing_interrupted()
        chunk = audio_48k[:, i:i + piece_samples].to(isolator.device)
        with torch.inference_mode():
            out_piece = session.push(chunk).cpu()
        pieces.append(out_piece)
        if pbar:
            pbar.update(1)
        if (i // piece_samples + 1) % 50 == 0 and isolator.device.type == "cuda":
            comfy.model_management.soft_empty_cache()

    with torch.inference_mode():
        flush_piece = session.flush().cpu()
    pieces.append(flush_piece)
    if pbar:
        pbar.update(1)

    out_48k = torch.cat(pieces, dim=1)

    if target_sr != 48000:
        out_final = comfy.audio.resample(out_48k, 48000, target_sr).squeeze(0)
    else:
        out_final = out_48k.squeeze(0)

    if out_final.shape[-1] > target_len:
        out_final = out_final[:target_len]
    elif out_final.shape[-1] < target_len:
        out_final = torch.nn.functional.pad(out_final, (0, target_len - out_final.shape[-1]))

    return out_final


class InworldVoiceIsolator:
    @classmethod
    def INPUT_TYPES(cls):
        model_files = _get_inworld_model_list()
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "mode": (["offline (high quality)", "streaming (realtime)"], {"default": "offline (high quality)"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "channel_handling": (["mono", "stereo (dual-channel)", "centered stereo"], {"default": "mono"}),
                "chunk_seconds": ("FLOAT", {"default": 8.0, "min": 2.0, "max": 30.0, "step": 1.0, "round": 0.1}),
                "overlap_seconds": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 8.0, "step": 0.5, "round": 0.1}),
                "output_sample_rate": (["original", "48000"], {"default": "original"}),
                "model_name": (model_files, {"default": "auto (download from HF if missing)"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("voice", "background")
    FUNCTION = "isolate_voice"
    CATEGORY = "audio/isolation"

    def isolate_voice(
        self,
        audio: dict,
        mode: str = "offline (high quality)",
        device: str = "auto",
        channel_handling: str = "mono",
        chunk_seconds: float = 8.0,
        overlap_seconds: float = 2.0,
        output_sample_rate: str = "original",
        model_name: str = "auto (download from HF if missing)",
    ):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)

        batch_size, num_channels, num_samples = waveform.shape
        model_target, resolved_mode = _resolve_model_and_mode(model_name, mode)
        isolator = _get_isolator(model_target, resolved_mode, device)
        target_sr = 48000 if output_sample_rate == "48000" else sample_rate
        is_offline = (resolved_mode == "offline")

        # Plan progress bar steps
        samples_48k = int(round(num_samples * 48000 / sample_rate))
        channels_to_process = 2 if (channel_handling == "stereo (dual-channel)" and num_channels >= 2) else 1

        if is_offline:
            chunk_samples, overlap_samples = window_samples(chunk_seconds, overlap_seconds, 48000, isolator.model.config.hop_length)
            if samples_48k <= chunk_samples:
                chunks_per_channel = 1
            else:
                starts = list(range(0, samples_48k - chunk_samples, chunk_samples - overlap_samples)) + [samples_48k - chunk_samples]
                chunks_per_channel = len(starts)
        else:
            piece_samples = int(0.5 * 48000)
            chunks_per_channel = (samples_48k + piece_samples - 1) // piece_samples + 1

        total_steps = batch_size * channels_to_process * chunks_per_channel
        pbar = ConsoleProgressBar(total_steps)

        def _run_isolation(mono_tensor):
            if is_offline:
                return _isolate_mono_offline(
                    isolator=isolator,
                    audio_mono=mono_tensor,
                    sample_rate=sample_rate,
                    target_sr=target_sr,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                    pbar=pbar,
                )
            else:
                return _isolate_mono_streaming(
                    isolator=isolator,
                    audio_mono=mono_tensor,
                    sample_rate=sample_rate,
                    target_sr=target_sr,
                    pbar=pbar,
                )

        def _calc_background(original_channel, v_channel):
            if target_sr == sample_rate:
                orig = original_channel.squeeze().cpu()
            else:
                orig = comfy.audio.resample(original_channel.unsqueeze(0), sample_rate, target_sr).squeeze().cpu()
            if orig.shape[-1] > v_channel.shape[-1]:
                orig = orig[:v_channel.shape[-1]]
            elif orig.shape[-1] < v_channel.shape[-1]:
                orig = torch.nn.functional.pad(orig, (0, v_channel.shape[-1] - orig.shape[-1]))
            return orig - v_channel

        voice_batches = []
        bg_batches = []

        for b in range(batch_size):
            item = waveform[b]  # [channels, samples]

            if channel_handling == "stereo (dual-channel)" and num_channels >= 2:
                v_chans = []
                b_chans = []
                for c in range(2):
                    ch = item[c].contiguous().cpu()
                    v = _run_isolation(ch)
                    bg = _calc_background(ch, v)
                    v_chans.append(v)
                    b_chans.append(bg)
                v_out = torch.stack(v_chans, dim=0)  # [2, samples]
                bg_out = torch.stack(b_chans, dim=0)
            else:
                mono = item.mean(dim=0).contiguous().cpu()
                v = _run_isolation(mono)
                bg = _calc_background(mono, v)

                if channel_handling == "centered stereo":
                    v_out = torch.stack([v, v], dim=0)  # [2, samples]
                    bg_out = torch.stack([bg, bg], dim=0)
                else:
                    v_out = v.unsqueeze(0)  # [1, samples]
                    bg_out = bg.unsqueeze(0)

            voice_batches.append(v_out)
            bg_batches.append(bg_out)

        out_voice = torch.stack(voice_batches, dim=0)
        out_bg = torch.stack(bg_batches, dim=0)

        return (
            {"waveform": out_voice, "sample_rate": target_sr},
            {"waveform": out_bg, "sample_rate": target_sr},
        )
