import unfake
import math
import torch
import numpy as np
from PIL import Image
import nest_asyncio
from .utils import unfake_scaledown_detect_pipeline
from comfy.utils import common_upscale
nest_asyncio.apply()


## ========Common Functions======== ##
def pillow_scale_up_nearest(img, target_resolution = 1024):
    
    R_MIN = int(target_resolution * 0.8)
    R_MAX = int(target_resolution * 1.6)
    
    w, h = img.size
    base = max(w, h)
    if base >= R_MAX:
        return img
    
    k_min = max(1, math.ceil(R_MIN / base))
    k_max = math.floor(R_MAX / base)
    if k_min > k_max:
        scale_factor = target_resolution / base
        new_size = (int(w * scale_factor), int(h * scale_factor))
        return img.resize(new_size, resample=Image.Resampling.NEAREST)
    
    best_k = min(range(k_min, k_max + 1), key=lambda k: abs(base * k - target_resolution))
    new_size = (w * best_k, h * best_k)

    return img.resize(new_size, resample=Image.Resampling.NEAREST)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2ndarray(image):
    return np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

def ndarray2tensor(image):
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)

def tensor_longest_side(image):
    height = image.shape[1]
    width = image.shape[2]
    return max(height, width)


## =========Node Classes========= ##
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
    CATEGORY = "BillBum/PixelTools"

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

class ForceDetectScale():

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "required": {
                "image": ("IMAGE",),
                "bypass_resolution": ("INT", {
                    "default": 256,
                    "min": 8,
                    "max": 4096,
                    "step": 8
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT",)
    RETURN_NAMES = ("down_img", "detected_scale")
    FUNCTION = "force_detect_scale"
    CATEGORY = "BillBum/PixelTools"

    def force_detect_scale(self, image, bypass_resolution):

        if tensor_longest_side(image) > bypass_resolution:

            np_img = tensor2ndarray(image)
            _, scale = unfake_scaledown_detect_pipeline(np_img, bypass_resolution)
            pil_img = Image.fromarray(np_img)
            original_width, original_height = pil_img.size
            new_width = int(original_width // scale)
            new_height = int(original_height // scale)
            upscaled_pil = pil_img.resize((new_width, new_height), Image.Resampling.NEAREST)
            upscaled_np = np.array(upscaled_pil)
            tensor_img = ndarray2tensor(upscaled_np)

        else:
            tensor_img = image
            scale = 1

        return (tensor_img, scale)

class ImageScaleDownByWH():

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": 4096, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": 4096, "step": 1, }),
            },
            "optional" : {
                "crop": (["disabled","center"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("out_img", "width", "height",)
    FUNCTION = "down_by_scale"
    CATEGORY = "BillBum/PixelTools"

    def down_by_scale(self, image, width, height, crop="disabled"):

        upscale_method="nearest-exact"
        
        image = image.movedim(-1,1)
        image = common_upscale(image, width, height, upscale_method, crop)
        image = image.movedim(1,-1)

        return(image, image.shape[2], image.shape[1],)

class ImageScaleDownByFactor():

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 1.0,"min": 0.0,"max": 100.0,"step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("out_img", "width", "height",)
    FUNCTION = "down_by_scale"
    CATEGORY = "BillBum/PixelTools"

    def down_by_scale(self, image, scale_factor):
        if scale_factor == 0:
            _, h, w, _ = image.shape
            return (image, w, h)

        output_images = []

        for img_tensor in image:

            pil_img = tensor2pil(img_tensor.unsqueeze(0))
            original_width, original_height = pil_img.size

            new_width, new_height = int(original_width // scale_factor), int(original_height // scale_factor)

            if new_width < 1 or new_height < 1:
                new_width = w
                new_height = h

            resized_pil = pil_img.resize((new_width, new_height), Image.Resampling.NEAREST)
            output_images.append(pil2tensor(resized_pil))

        tensor_out = torch.cat(output_images, dim=0)
        final_width, final_height = tensor_out.shape[3], tensor_out.shape[2]

        return(tensor_out, final_width, final_height)

class PixelUpscale2Target():

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "image": ("IMAGE", ),
                        "target_resolution": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 2}),
                    }
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale2target"
    CATEGORY = "BillBum/PixelTools"

    def upscale2target(self, image, target_resolution):

        processed_images = []
        for i in range(image.shape[0]):
            single_image_tensor = image[i:i+1]
            pil_image = tensor2pil(single_image_tensor)
            result_pil = pillow_scale_up_nearest(pil_image, target_resolution)
            result_tensor = pil2tensor(result_pil)
            processed_images.append(result_tensor)
            
        batch_output = torch.cat(processed_images, dim=0)
        return (batch_output,)
