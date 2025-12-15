import os
from .clip_encoder import CLIPVisionTower
from .imagebind import ImageBindWrapper
from .open_clip_encoder import OpenCLIPVisionTower
from .hf_vision import HFVisionTower
from .siglip_encoder import SigLipVisionTower
from .siglip2_anyres_encoder import SigLip2VisionTower
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .navit_encoder import NaViTVisionTower
from .temporal_encoder import TemporalEncoder, SpatialTemporalEncoder
from .tff import TemporalFiberFusion
# from .eva_clip.eva_clip_encoder import EvaClipVisionTower
# from .dev_eva_clip.eva_vit import EvaViTWrapper


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
    # is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, "s2", False)
    if vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "naflex" in vision_tower.lower():
        return SigLip2VisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif "siglip" in vision_tower.lower():
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif "navit" in vision_tower:
        return NaViTVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("hf:"):
        return HFVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower in ["imagebind_huge"]:
        return ImageBindWrapper(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("open_clip_hub"):
        return OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # elif "internal-eva" in vision_tower.lower() or "eva02" in vision_tower.lower():
    #     return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    # elif vision_tower in ["EVA-CLIP-8B", "EVA-CLIP-8B-plus"]:
    #     return EvaViTWrapper(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower}")



def build_temporal_encoder(config):
    temporal_encoder_type = getattr(config, "temporal_encoder_type", "temporal_only")
    temporal_layers = getattr(config, "temporal_layers", 2)
    if temporal_encoder_type == "temporal_only":
        return TemporalEncoder(temporal_layers=temporal_layers)
    elif temporal_encoder_type == "spatial_and_temporal":
        return SpatialTemporalEncoder(temporal_layers=temporal_layers)
    elif temporal_encoder_type == "tff":
        return TemporalFiberFusion(depth=temporal_layers)
    else:
        raise ValueError(f"Unknown temporal encoder type: {temporal_encoder_type}")
