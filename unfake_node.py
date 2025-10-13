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
    img = Image.fromarray(image_np)
    return img

def _pil_to_tensor(images: Image.Image | list[Image.Image]) -> torch.Tensor:
    """Converts a PIL Image or a list of PIL Images to a tensor."""
    def single_pil2tensor(image: Image.Image) -> torch.Tensor:
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:  # Grayscale
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W)
        else:  # RGB or RGBA
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W, C)

    if isinstance(images, Image.Image):
        return single_pil2tensor(images)
    else:
        return torch.cat([single_pil2tensor(img) for img in images], dim=0)

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
                "cleanup_morph": ("BOOLEAN", {"default": False}),
                "cleanup_jaggies": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "unfaked_image"
    CATEGORY = "BillBum/Image"

    def unfaked_image(self, image, max_colors, detect_method, downscale_method, cleanup_morph, cleanup_jaggies):
        
        pil_img = _tensor_to_pil(image)

        if max_colors == 0:
            result = unfake.process_image_sync(
                pil_img,
                detect_method=detect_method,
                downscale_method=downscale_method,
                cleanup={"morph": cleanup_morph, "jaggy": cleanup_jaggies},
                snap_grid=True
            )
        else:
            result = unfake.process_image_sync(
                pil_img,
                max_colors=max_colors,
                detect_method=detect_method,
                downscale_method=downscale_method,
                cleanup={"morph": cleanup_morph, "jaggy": cleanup_jaggies},
                snap_grid=True
            )
        
        result_tensor = _pil_to_tensor(result["image"])
        return (result_tensor,)



NODE_CLASS_MAPPINGS = {
    "CustomUnfake": CustomUnfake,
}

NODE_DISPLAY_NAME_MAPPINGS = {
}