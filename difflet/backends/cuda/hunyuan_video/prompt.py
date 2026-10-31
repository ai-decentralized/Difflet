"""Torch-free prompt constants shared by the HunyuanVideo GPU spec and application.

The template matches the existing HunyuanVideo CLI and Diffusers 0.38.0.
"""

LLAMA_TEMPLATE = "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
LLAMA_CROP_START = 95
TEXT_SEQ_LEN = 256
