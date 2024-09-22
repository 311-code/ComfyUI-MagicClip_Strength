# Please support 311_code on https://ko-fi.com/311_code

import torch
from nodes import MAX_RESOLUTION

class CLIPTextEncodeSDXL_311_code:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text_g": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP", ),
            "text_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip": ("CLIP", ),
            "clip_g_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "clip_l_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "size_cond_factor": ("INT", {"default": 4, "min": 1, "max": 16}),
            "layer_idx": ("INT", {"default": -2, "min": -33, "max": 33}),  # Added layer_idx
            }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"
    CATEGORY = "essentials"

    def execute(self, clip, width, height, size_cond_factor, text_g, text_l, clip_g_strength, clip_l_strength, layer_idx):
        width = width * size_cond_factor
        height = height * size_cond_factor

        cond_combined = None
        pooled_combined = None

        # Modify CLIP layers based on `layer_idx`
        clip = clip.clone()  # Clone the clip model to ensure it's modifiable
        clip.clip_layer(layer_idx)  # Set the stopping point for the CLIP model based on layer_idx

        # Tokenize both texts, or use empty tokens if both are empty
        tokens_g = clip.tokenize(text_g) if text_g.strip() else clip.tokenize("")
        tokens_l = clip.tokenize(text_l) if text_l.strip() else clip.tokenize("")

        # Ensure equal length by padding with empty tokens if needed
        if len(tokens_l["l"]) != len(tokens_g["g"]):
            empty_g = clip.tokenize("")["g"]
            empty_l = clip.tokenize("")["l"]
            while len(tokens_l["l"]) < len(tokens_g["g"]):
                tokens_l["l"] += empty_l
            while len(tokens_g["g"]) < len(tokens_l["l"]):
                tokens_g["g"] += empty_g

        # Encode tokens
        tokens = {"g": tokens_g["g"], "l": tokens_l["l"]}
        cond_combined, pooled_combined = clip.encode_from_tokens(tokens, return_pooled=True)

        # Apply strengths to g and l if needed
        if cond_combined.dim() == 3:
            if text_g.strip():  # Only apply strength to g if text_g is not empty
                cond_combined[:, :, :] *= clip_g_strength
            if text_l.strip():  # Only apply strength to l if text_l is not empty
                cond_combined[:, :, :] *= clip_l_strength

        # Return the conditioning even if both texts were empty
        return ([[cond_combined, {"pooled_output": pooled_combined, "width": width, "height": height}]], )


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeSDXL_311_code": CLIPTextEncodeSDXL_311_code
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeSDXL_311_code": "311_code SDXL Clip Text Encode"
}
           
