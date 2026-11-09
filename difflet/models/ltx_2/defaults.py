"""LTX-2 default request shape and text length, shared by every backend.

Torch-free: the GPU spec (``spec.py``) reads these in ``difflet serve``'s parent process.
``contract.py`` re-exports them for existing importers.
"""

LTX_2_DEFAULT_TEXT_SEQ_LEN = 1024
LTX_2_DEFAULT_HEIGHT = 512
LTX_2_DEFAULT_WIDTH = 768
LTX_2_DEFAULT_NUM_FRAMES = 121

__all__ = [
    "LTX_2_DEFAULT_HEIGHT",
    "LTX_2_DEFAULT_NUM_FRAMES",
    "LTX_2_DEFAULT_TEXT_SEQ_LEN",
    "LTX_2_DEFAULT_WIDTH",
]
