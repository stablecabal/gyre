try:
    from diffusers import ControlNetModel
except ImportError:
    print("Falling back to internal ControlNetModel")
    from .models import ControlNetModel

from .unet_patcher import patch_unet, unpatch_unet
