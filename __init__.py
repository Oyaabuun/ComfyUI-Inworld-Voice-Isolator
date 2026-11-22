from .nodes import InworldVoiceIsolator, InworldSaveAudio

NODE_CLASS_MAPPINGS = {
    "InworldVoiceIsolator": InworldVoiceIsolator,
    "InworldSaveAudio": InworldSaveAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InworldVoiceIsolator": "Inworld Voice Isolator",
    "InworldSaveAudio": "Save Audio (Multi-Format)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
