import unfake
import torch
import numpy as np
from PIL import Image
import nest_asyncio
nest_asyncio.apply()

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class CustomUnfake():
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "required": {
                "image": ("IMAGE",),
                "max_colors": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "display": "number"
                }),
                "detect_method": (["auto", "runs", "edge"], {"default": "auto"}),
                "downscale_method": (["dominant", "median", "mode", "mean", "content-adaptive"], {"default": "dominant"}),
                "cleanup_morph": ("BOOLEAN", {"default": True}),
                "cleanup_jaggies": ("BOOLEAN", {"default": False}),
                "transparent_background": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE","STRING")
    RETURN_NAMES = ("align_img", "align_info")
    FUNCTION = "unfaked_image"
    CATEGORY = "BillBum/Image"

    def unfaked_image(self, image, max_colors, detect_method, downscale_method, cleanup_morph, cleanup_jaggies, transparent_background):
        
        pil_img = tensor2pil(image)

        if max_colors == 0:
            result = unfake.process_image_sync(
                pil_img,
                detect_method=detect_method,
                downscale_method=downscale_method,
                cleanup={"morph": cleanup_morph, "jaggy": cleanup_jaggies},
                snap_grid=True,
                transparent_background=transparent_background,
                auto_color_detect=True
            )
        else:
            result = unfake.process_image_sync(
                pil_img,
                max_colors=max_colors,
                detect_method=detect_method,
                downscale_method=downscale_method,
                cleanup={"morph": cleanup_morph, "jaggy": cleanup_jaggies},
                snap_grid=True,
                transparent_background=transparent_background
            )
        manifest = result["manifest"]
        width, height = manifest.final_size
        final_size = f"{width}x{height}"
        final_colors = manifest.processing_steps["color_quantization"]["final_colors"]
        out_info = f"{final_size}px_c{final_colors}"
        result_tensor = pil2tensor(result["image"])
        return (result_tensor, out_info)
