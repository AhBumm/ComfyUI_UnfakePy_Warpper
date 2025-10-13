import unfake
import torch
import numpy as np
from PIL import Image
import nest_asyncio
nest_asyncio.apply()

def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if len(image.shape) > 3:
        image = image[0]
    input_tensor = image.cpu()
    image_np = (input_tensor.numpy() * 255).astype(np.uint8)
    if image_np.shape[-1] == 4:
        mode = "RGBA"
    elif image_np.shape[-1] == 3:
        mode = "RGB"
    else:
        mode = None
    return Image.fromarray(image_np, mode=mode)

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    np_image = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(np_image).unsqueeze(0)

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
        
        pil_img = _tensor_to_pil(image)

        if max_colors == 0:
            result = unfake.process_image_sync(
                pil_img,
                detect_method=detect_method,
                downscale_method=downscale_method,
                cleanup={"morph": cleanup_morph, "jaggy": cleanup_jaggies},
                snap_grid=True,
                transparent_background=transparent_background
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
        result_tensor = _pil_to_tensor(result["image"])
        return (result_tensor, out_info)

## Meneger Mapping
NODE_CLASS_MAPPINGS = {
    "CustomUnfake": CustomUnfake,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}