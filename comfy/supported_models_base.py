from . import model_base, latent_formats


class ClipTarget:
    def __init__(self, tokenizer, clip):
        self.clip = clip
        self.tokenizer = tokenizer
        self.params = {}

class BASE:
    unet_config = {}
    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
    }
    clip_vision_prefix = None
    noise_aug_config = None
    beta_schedule = "linear"
    latent_format = latent_formats.LatentFormat

    @classmethod
    def matches(s, unet_config):
        for k in s.unet_config:
            if s.unet_config[k] != unet_config[k]:
                return False
        return True

    def model_type(self, state_dict, prefix=""):
        return model_base.EPS

    def inpaint_model(self):
        return self.unet_config["in_channels"] > 4

    def __init__(self, unet_config):
        self.unet_config = unet_config
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def get_model(self, state_dict, prefix="", device=None):
        out = model_base.BaseModel(self, model_type=self.model_type(state_dict, prefix), device=device)
        if self.inpaint_model():
            out.set_inpaint()
        return out