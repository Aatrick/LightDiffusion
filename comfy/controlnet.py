import math

import torch

import comfy.adapter
import comfy.cldm.cldm
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.utils as utils


def broadcast_image_to(tensor, target_batch_size, batched_number):
    current_batch_size = tensor.shape[0]
    #print(current_batch_size, target_batch_size)
    if current_batch_size == 1:
        return tensor

    per_batch = target_batch_size // batched_number
    tensor = tensor[:per_batch]

    if per_batch > tensor.shape[0]:
        tensor = torch.cat([tensor] * (per_batch // tensor.shape[0]) + [tensor[:(per_batch % tensor.shape[0])]], dim=0)

    current_batch_size = tensor.shape[0]
    if current_batch_size == target_batch_size:
        return tensor
    else:
        return torch.cat([tensor] * batched_number, dim=0)

class ControlBase:
    def __init__(self, device=None):
        self.cond_hint_original = None
        self.cond_hint = None
        self.strength = 1.0
        self.timestep_percent_range = (1.0, 0.0)
        self.timestep_range = None

        if device is None:
            device = comfy.model_management.get_torch_device()
        self.device = device
        self.previous_controlnet = None
        self.global_average_pooling = False

    def pre_run(self, model, percent_to_timestep_function):
        self.timestep_range = (percent_to_timestep_function(self.timestep_percent_range[0]), percent_to_timestep_function(self.timestep_percent_range[1]))
        if self.previous_controlnet is not None:
            self.previous_controlnet.pre_run(model, percent_to_timestep_function)

    def cleanup(self):
        if self.previous_controlnet is not None:
            self.previous_controlnet.cleanup()
        if self.cond_hint is not None:
            del self.cond_hint
            self.cond_hint = None
        self.timestep_range = None

    def get_models(self):
        out = []
        if self.previous_controlnet is not None:
            out += self.previous_controlnet.get_models()
        return out

    def copy_to(self, c):
        c.cond_hint_original = self.cond_hint_original
        c.strength = self.strength
        c.timestep_percent_range = self.timestep_percent_range

    def inference_memory_requirements(self, dtype):
        if self.previous_controlnet is not None:
            return self.previous_controlnet.inference_memory_requirements(dtype)
        return 0

    def control_merge(self, control_input, control_output, control_prev, output_dtype):
        out = {'input':[], 'middle':[], 'output': []}

        if control_input is not None:
            for i in range(len(control_input)):
                key = 'input'
                x = control_input[i]
                if x is not None:
                    x *= self.strength
                    if x.dtype != output_dtype:
                        x = x.to(output_dtype)
                out[key].insert(0, x)

        if control_output is not None:
            for i in range(len(control_output)):
                if i == (len(control_output) - 1):
                    key = 'middle'
                    index = 0
                else:
                    key = 'output'
                    index = i
                x = control_output[i]
                if x is not None:
                    if self.global_average_pooling:
                        x = torch.mean(x, dim=(2, 3), keepdim=True).repeat(1, 1, x.shape[2], x.shape[3])

                    x *= self.strength
                    if x.dtype != output_dtype:
                        x = x.to(output_dtype)

                out[key].append(x)
        if control_prev is not None:
            for x in ['input', 'middle', 'output']:
                o = out[x]
                for i in range(len(control_prev[x])):
                    prev_val = control_prev[x][i]
                    if i >= len(o):
                        o.append(prev_val)
                    elif prev_val is not None:
                        if o[i] is None:
                            o[i] = prev_val
                        else:
                            o[i] += prev_val
        return out

class ControlNet(ControlBase):
    def __init__(self, control_model, global_average_pooling=False, device=None):
        super().__init__(device)
        self.control_model = control_model
        self.control_model_wrapped = comfy.model_patcher.ModelPatcher(self.control_model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.unet_offload_device())
        self.global_average_pooling = global_average_pooling
        self.model_sampling_current = None

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        output_dtype = x_noisy.dtype
        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = None
            self.cond_hint = utils.common_upscale(self.cond_hint_original, x_noisy.shape[3] * 8, x_noisy.shape[2] * 8, 'nearest-exact', "center").to(self.control_model.dtype).to(self.device)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)


        context = cond['c_crossattn']
        y = cond.get('y', None)
        if y is not None:
            y = y.to(self.control_model.dtype)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        control = self.control_model(x=x_noisy.to(self.control_model.dtype), hint=self.cond_hint, timesteps=timestep.float(), context=context.to(self.control_model.dtype), y=y)
        return self.control_merge(None, control, control_prev, output_dtype)

    def copy(self):
        c = ControlNet(self.control_model, global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        return c

    def get_models(self):
        out = super().get_models()
        out.append(self.control_model_wrapped)
        return out

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        self.model_sampling_current = model.model_sampling

    def cleanup(self):
        self.model_sampling_current = None
        super().cleanup()

class ControlLoraOps:
    class Linear(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.up = None
            self.down = None
            self.bias = None

    class Conv2d(torch.nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            device=None,
            dtype=None
        ):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.padding_mode = padding_mode

            self.weight = None
            self.bias = None
            self.up = None
            self.down = None

    def conv_nd(self, dims, *args, **kwargs):
        if dims == 2:
            return self.Conv2d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class ControlLora(ControlNet):
    def __init__(self, control_weights, global_average_pooling=False, device=None):
        ControlBase.__init__(self, device)
        self.control_weights = control_weights
        self.global_average_pooling = global_average_pooling

    def pre_run(self, model, percent_to_timestep_function):
        super().pre_run(model, percent_to_timestep_function)
        controlnet_config = model.model_config.unet_config.copy()
        controlnet_config.pop("out_channels")
        controlnet_config["hint_channels"] = self.control_weights["input_hint_block.0.weight"].shape[1]
        controlnet_config["operations"] = ControlLoraOps()
        self.control_model = comfy.cldm.cldm.ControlNet(**controlnet_config)
        dtype = model.get_dtype()
        self.control_model.to(dtype)
        self.control_model.to(comfy.model_management.get_torch_device())
        diffusion_model = model.diffusion_model
        sd = diffusion_model.state_dict()

        for k in sd:
            weight = comfy.model_management.resolve_lowvram_weight(sd[k], diffusion_model, k)
            try:
                utils.set_attr(self.control_model, k, weight)
            except:
                pass

        for k in self.control_weights:
            if k not in {"lora_controlnet"}:
                utils.set_attr(self.control_model, k, self.control_weights[k].to(dtype).to(comfy.model_management.get_torch_device()))

    def copy(self):
        c = ControlLora(self.control_weights, global_average_pooling=self.global_average_pooling)
        self.copy_to(c)
        return c

    def cleanup(self):
        del self.control_model
        self.control_model = None
        super().cleanup()

    def get_models(self):
        out = ControlBase.get_models(self)
        return out

    def inference_memory_requirements(self, dtype):
        return utils.calculate_parameters(self.control_weights) * comfy.model_management.dtype_size(dtype) + ControlBase.inference_memory_requirements(self, dtype)


class T2IAdapter(ControlBase):
    def __init__(self, t2i_model, channels_in, device=None):
        super().__init__(device)
        self.t2i_model = t2i_model
        self.channels_in = channels_in
        self.control_input = None

    def scale_image_to(self, width, height):
        unshuffle_amount = self.t2i_model.unshuffle_amount
        width = math.ceil(width / unshuffle_amount) * unshuffle_amount
        height = math.ceil(height / unshuffle_amount) * unshuffle_amount
        return width, height

    def get_control(self, x_noisy, t, cond, batched_number):
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        if self.cond_hint is None or x_noisy.shape[2] * 8 != self.cond_hint.shape[2] or x_noisy.shape[3] * 8 != self.cond_hint.shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.control_input = None
            self.cond_hint = None
            width, height = self.scale_image_to(x_noisy.shape[3] * 8, x_noisy.shape[2] * 8)
            self.cond_hint = utils.common_upscale(self.cond_hint_original, width, height, 'nearest-exact', "center").float().to(self.device)
            if self.channels_in == 1 and self.cond_hint.shape[1] > 1:
                self.cond_hint = torch.mean(self.cond_hint, 1, keepdim=True)
        if x_noisy.shape[0] != self.cond_hint.shape[0]:
            self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)
        if self.control_input is None:
            self.t2i_model.to(x_noisy.dtype)
            self.t2i_model.to(self.device)
            self.control_input = self.t2i_model(self.cond_hint.to(x_noisy.dtype))
            self.t2i_model.cpu()

        control_input = list(map(lambda a: None if a is None else a.clone(), self.control_input))
        mid = None
        if self.t2i_model.xl == True:
            mid = control_input[-1:]
            control_input = control_input[:-1]
        return self.control_merge(control_input, mid, control_prev, x_noisy.dtype)

    def copy(self):
        c = T2IAdapter(self.t2i_model, self.channels_in)
        self.copy_to(c)
        return c