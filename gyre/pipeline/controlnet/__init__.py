try:
    from diffusers import ControlNetModel
except ImportError:
    from .models import ControlNetModel

from .unet_patcher import patch_unet, unpatch_unet
