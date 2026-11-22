from .nodes import InworldVoiceIsolator

NODE_CLASS_MAPPINGS = {
    "InworldVoiceIsolator": InworldVoiceIsolator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InworldVoiceIsolator": "Inworld Voice Isolator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
