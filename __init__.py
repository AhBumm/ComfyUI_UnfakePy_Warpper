from .unfake_node import *

## Meneger Mapping
NODE_CLASS_MAPPINGS = {
    "Unfake_PixelateTools": CustomUnfake,
    "ForceDetectPixelateScale": ForceDetectScale,
    "NearestImageScaleDown": ImageScaleDownByWH,
    "ImageScaleDownByFactor": ImageScaleDownByFactor,
    "PixelUpscale2Target": PixelUpscale2Target,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Unfake_PixelateTools": "Unfake Pixelate Tools",
    "ForceDetectPixelateScale": "Force Detect Pixelate Scale",
    "NearestImageScaleDown": "Image Downscale By W&H",
    "ImageScaleDownByFactor": "Image Downscale By Factor",
    "PixelUpscale2Target": "PixelIMG Upscale To Target",
}
