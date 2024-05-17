def torch_gc():
    pass

from PIL import Image


def flatten(img, bgcolor):
    # Replace transparency with bgcolor
    if img.mode in ("RGB"):
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img).convert("RGB")

class Script:
    pass

class Options:
    img2img_background_color = "#ffffff"  # Set to white for now


class State:
    interrupted = False

    def begin(self):
        pass

    def end(self):
        pass


opts = Options()
state = State()

# Will only ever hold 1 upscaler
sd_upscalers = [None]
# The upscaler usable by ComfyUI nodes
actual_upscaler = None

# Batch of images to upscale
batch = None


import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image

BLUR_KERNEL_SIZE = 15


def tensor_to_pil(img_tensor, batch_index=0):
    # Takes an image in a batch in the form of a tensor of shape [batch_size, channels, height, width]
    # and returns an PIL Image with the corresponding mode deduced by the number of channels

    # Take the image in the batch given by batch_index
    img_tensor = img_tensor[batch_index].unsqueeze(0)
    i = 255. * img_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image


def controlnet_hint_to_pil(tensor, batch_index=0):
    return tensor_to_pil(tensor.movedim(1, -1), batch_index)


def pil_to_controlnet_hint(img):
    return pil_to_tensor(img).movedim(-1, 1)


def crop_tensor(tensor, region):
    # Takes a tensor of shape [batch_size, height, width, channels] and crops it to the given region
    x1, y1, x2, y2 = region
    return tensor[:, y1:y2, x1:x2, :]


def resize_tensor(tensor, size, mode="nearest-exact"):
    # Takes a tensor of shape [B, C, H, W] and resizes
    # it to a shape of [B, C, size[0], size[1]] using the given mode
    return torch.nn.functional.interpolate(tensor, size=size, mode=mode)


def get_crop_region(mask, pad=0):
    # Takes a black and white PIL image in 'L' mode and returns the coordinates of the white rectangular mask region
    # Should be equivalent to the get_crop_region function from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/masking.py
    coordinates = mask.getbbox()
    if coordinates is not None:
        x1, y1, x2, y2 = coordinates
    else:
        x1, y1, x2, y2 = mask.width, mask.height, 0, 0
    # Apply padding
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, mask.width)
    y2 = min(y2 + pad, mask.height)
    return fix_crop_region((x1, y1, x2, y2), (mask.width, mask.height))


def fix_crop_region(region, image_size):
    # Remove the extra pixel added by the get_crop_region function
    image_width, image_height = image_size
    x1, y1, x2, y2 = region
    if x2 < image_width:
        x2 -= 1
    if y2 < image_height:
        y2 -= 1
    return x1, y1, x2, y2


def expand_crop(region, width, height, target_width, target_height):
    '''
    Expands a crop region to a specified target size.
    :param region: A tuple of the form (x1, y1, x2, y2) denoting the upper left and the lower right points
        of the rectangular region. Expected to have x2 > x1 and y2 > y1.
    :param width: The width of the image the crop region is from.
    :param height: The height of the image the crop region is from.
    :param target_width: The desired width of the crop region.
    :param target_height: The desired height of the crop region.
    '''
    x1, y1, x2, y2 = region
    actual_width = x2 - x1
    actual_height = y2 - y1
    # target_width = math.ceil(actual_width / 8) * 8
    # target_height = math.ceil(actual_height / 8) * 8

    # Try to expand region to the right of half the difference
    width_diff = target_width - actual_width
    x2 = min(x2 + width_diff // 2, width)
    # Expand region to the left of the difference including the pixels that could not be expanded to the right
    width_diff = target_width - (x2 - x1)
    x1 = max(x1 - width_diff, 0)
    # Try the right again
    width_diff = target_width - (x2 - x1)
    x2 = min(x2 + width_diff, width)

    # Try to expand region to the bottom of half the difference
    height_diff = target_height - actual_height
    y2 = min(y2 + height_diff // 2, height)
    # Expand region to the top of the difference including the pixels that could not be expanded to the bottom
    height_diff = target_height - (y2 - y1)
    y1 = max(y1 - height_diff, 0)
    # Try the bottom again
    height_diff = target_height - (y2 - y1)
    y2 = min(y2 + height_diff, height)

    return (x1, y1, x2, y2), (target_width, target_height)


def resize_region(region, init_size, resize_size):
    # Resize a crop so that it fits an image that was resized to the given width and height
    x1, y1, x2, y2 = region
    init_width, init_height = init_size
    resize_width, resize_height = resize_size
    x1 = math.floor(x1 * resize_width / init_width)
    x2 = math.ceil(x2 * resize_width / init_width)
    y1 = math.floor(y1 * resize_height / init_height)
    y2 = math.ceil(y2 * resize_height / init_height)
    return (x1, y1, x2, y2)


def pad_image(image, left_pad, right_pad, top_pad, bottom_pad, fill=False, blur=False):
    '''
    Pads an image with the given number of pixels on each side and fills the padding with data from the edges.
    :param image: A PIL image
    :param left_pad: The number of pixels to pad on the left side
    :param right_pad: The number of pixels to pad on the right side
    :param top_pad: The number of pixels to pad on the top side
    :param bottom_pad: The number of pixels to pad on the bottom side
    :param blur: Whether to blur the padded edges
    :return: A PIL image with size (image.width + left_pad + right_pad, image.height + top_pad + bottom_pad)
    '''
    left_edge = image.crop((0, 1, 1, image.height - 1))
    right_edge = image.crop((image.width - 1, 1, image.width, image.height - 1))
    top_edge = image.crop((1, 0, image.width - 1, 1))
    bottom_edge = image.crop((1, image.height - 1, image.width - 1, image.height))
    new_width = image.width + left_pad + right_pad
    new_height = image.height + top_pad + bottom_pad
    padded_image = Image.new(image.mode, (new_width, new_height))
    padded_image.paste(image, (left_pad, top_pad))
    if fill:
        for i in range(left_pad):
            edge = left_edge.resize(
                (1, new_height - i * (top_pad + bottom_pad) // left_pad), resample=Image.Resampling.NEAREST)
            padded_image.paste(edge, (i, i * top_pad // left_pad))
        for i in range(right_pad):
            edge = right_edge.resize(
                (1, new_height - i * (top_pad + bottom_pad) // right_pad), resample=Image.Resampling.NEAREST)
            padded_image.paste(edge, (new_width - 1 - i, i * top_pad // right_pad))
        for i in range(top_pad):
            edge = top_edge.resize(
                (new_width - i * (left_pad + right_pad) // top_pad, 1), resample=Image.Resampling.NEAREST)
            padded_image.paste(edge, (i * left_pad // top_pad, i))
        for i in range(bottom_pad):
            edge = bottom_edge.resize(
                (new_width - i * (left_pad + right_pad) // bottom_pad, 1), resample=Image.Resampling.NEAREST)
            padded_image.paste(edge, (i * left_pad // bottom_pad, new_height - 1 - i))
        if blur and not (left_pad == right_pad == top_pad == bottom_pad == 0):
            padded_image = padded_image.filter(ImageFilter.GaussianBlur(BLUR_KERNEL_SIZE))
            padded_image.paste(image, (left_pad, top_pad))
    return padded_image


def pad_image2(image, left_pad, right_pad, top_pad, bottom_pad, fill=False, blur=False):
    '''
    Pads an image with the given number of pixels on each side and fills the padding with data from the edges.
    Faster than pad_image, but only pads with edge data in straight lines.
    :param image: A PIL image
    :param left_pad: The number of pixels to pad on the left side
    :param right_pad: The number of pixels to pad on the right side
    :param top_pad: The number of pixels to pad on the top side
    :param bottom_pad: The number of pixels to pad on the bottom side
    :param blur: Whether to blur the padded edges
    :return: A PIL image with size (image.width + left_pad + right_pad, image.height + top_pad + bottom_pad)
    '''
    left_edge = image.crop((0, 1, 1, image.height - 1))
    right_edge = image.crop((image.width - 1, 1, image.width, image.height - 1))
    top_edge = image.crop((1, 0, image.width - 1, 1))
    bottom_edge = image.crop((1, image.height - 1, image.width - 1, image.height))
    new_width = image.width + left_pad + right_pad
    new_height = image.height + top_pad + bottom_pad
    padded_image = Image.new(image.mode, (new_width, new_height))
    padded_image.paste(image, (left_pad, top_pad))
    if fill:
        if left_pad > 0:
            padded_image.paste(left_edge.resize((left_pad, new_height), resample=Image.Resampling.NEAREST), (0, 0))
        if right_pad > 0:
            padded_image.paste(right_edge.resize((right_pad, new_height),
                                                 resample=Image.Resampling.NEAREST), (new_width - right_pad, 0))
        if top_pad > 0:
            padded_image.paste(top_edge.resize((new_width, top_pad), resample=Image.Resampling.NEAREST), (0, 0))
        if bottom_pad > 0:
            padded_image.paste(bottom_edge.resize((new_width, bottom_pad),
                                                  resample=Image.Resampling.NEAREST), (0, new_height - bottom_pad))
        if blur and not (left_pad == right_pad == top_pad == bottom_pad == 0):
            padded_image = padded_image.filter(ImageFilter.GaussianBlur(BLUR_KERNEL_SIZE))
            padded_image.paste(image, (left_pad, top_pad))
    return padded_image


def pad_tensor(tensor, left_pad, right_pad, top_pad, bottom_pad, fill=False, blur=False):
    '''
    Pads an image tensor with the given number of pixels on each side and fills the padding with data from the edges.
    :param tensor: A tensor of shape [B, H, W, C]
    :param left_pad: The number of pixels to pad on the left side
    :param right_pad: The number of pixels to pad on the right side
    :param top_pad: The number of pixels to pad on the top side
    :param bottom_pad: The number of pixels to pad on the bottom side
    :param blur: Whether to blur the padded edges
    :return: A tensor of shape [B, H + top_pad + bottom_pad, W + left_pad + right_pad, C]
    '''
    batch_size, channels, height, width = tensor.shape
    h_pad = left_pad + right_pad
    v_pad = top_pad + bottom_pad
    new_width = width + h_pad
    new_height = height + v_pad

    # Create empty image
    padded = torch.zeros((batch_size, channels, new_height, new_width), dtype=tensor.dtype)

    # Copy the original image into the centor of the padded tensor
    padded[:, :, top_pad:top_pad + height, left_pad:left_pad + width] = tensor

    # Duplicate the edges of the original image into the padding
    if top_pad > 0:
        padded[:, :, :top_pad, :] = padded[:, :, top_pad:top_pad + 1, :]  # Top edge
    if bottom_pad > 0:
        padded[:, :, -bottom_pad:, :] = padded[:, :, -bottom_pad - 1:-bottom_pad, :]  # Bottom edge
    if left_pad > 0:
        padded[:, :, :, :left_pad] = padded[:, :, :, left_pad:left_pad + 1]  # Left edge
    if right_pad > 0:
        padded[:, :, :, -right_pad:] = padded[:, :, :, -right_pad - 1:-right_pad]  # Right edge

    return padded


def resize_and_pad_image(image, width, height, fill=False, blur=False):
    '''
    Resizes an image to the given width and height and pads it to the given width and height.
    :param image: A PIL image
    :param width: The width of the resized image
    :param height: The height of the resized image
    :param fill: Whether to fill the padding with data from the edges
    :param blur: Whether to blur the padded edges
    :return: A PIL image of size (width, height)
    '''
    width_ratio = width / image.width
    height_ratio = height / image.height
    if height_ratio > width_ratio:
        resize_ratio = width_ratio
    else:
        resize_ratio = height_ratio
    resize_width = round(image.width * resize_ratio)
    resize_height = round(image.height * resize_ratio)
    resized = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)
    # Pad the sides of the image to get the image to the desired size that wasn't covered by the resize
    horizontal_pad = (width - resize_width) // 2
    vertical_pad = (height - resize_height) // 2
    result = pad_image2(resized, horizontal_pad, horizontal_pad, vertical_pad, vertical_pad, fill, blur)
    result = result.resize((width, height), resample=Image.Resampling.LANCZOS)
    return result, (horizontal_pad, vertical_pad)


def resize_and_pad_tensor(tensor, width, height, fill=False, blur=False):
    '''
    Resizes an image tensor to the given width and height and pads it to the given width and height.
    :param tensor: A tensor of shape [B, H, W, C]
    :param width: The width of the resized image
    :param height: The height of the resized image
    :param fill: Whether to fill the padding with data from the edges
    :param blur: Whether to blur the padded edges
    :return: A tensor of shape [B, height, width, C]
    '''
    # Resize the image to the closest size that maintains the aspect ratio
    width_ratio = width / tensor.shape[3]
    height_ratio = height / tensor.shape[2]
    if height_ratio > width_ratio:
        resize_ratio = width_ratio
    else:
        resize_ratio = height_ratio
    resize_width = round(tensor.shape[3] * resize_ratio)
    resize_height = round(tensor.shape[2] * resize_ratio)
    resized = F.interpolate(tensor, size=(resize_height, resize_width), mode='nearest-exact')
    # Pad the sides of the image to get the image to the desired size that wasn't covered by the resize
    horizontal_pad = (width - resize_width) // 2
    vertical_pad = (height - resize_height) // 2
    result = pad_tensor(resized, horizontal_pad, horizontal_pad, vertical_pad, vertical_pad, fill, blur)
    result = F.interpolate(result, size=(height, width), mode='nearest-exact')
    return result


def crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    if "control" not in cond_dict:
        return
    c = cond_dict["control"]
    controlnet = c.copy()
    cond_dict["control"] = controlnet
    while c is not None:
        # hint is shape (B, C, H, W)
        hint = controlnet.cond_hint_original
        resized_crop = resize_region(region, canvas_size, hint.shape[:-3:-1])
        hint = crop_tensor(hint.movedim(1, -1), resized_crop).movedim(-1, 1)
        hint = resize_tensor(hint, tile_size[::-1])
        controlnet.cond_hint_original = hint
        c = c.previous_controlnet
        controlnet.set_previous_controlnet(c.copy() if c is not None else None)
        controlnet = controlnet.previous_controlnet


def region_intersection(region1, region2):
    """
    Returns the coordinates of the intersection of two rectangular regions.
    :param region1: A tuple of the form (x1, y1, x2, y2) denoting the upper left and the lower right points
        of the first rectangular region. Expected to have x2 > x1 and y2 > y1.
    :param region2: The second rectangular region with the same format as the first.
    :return: A tuple of the form (x1, y1, x2, y2) denoting the rectangular intersection.
        None if there is no intersection.
    """
    x1, y1, x2, y2 = region1
    x1_, y1_, x2_, y2_ = region2
    x1 = max(x1, x1_)
    y1 = max(y1, y1_)
    x2 = min(x2, x2_)
    y2 = min(y2, y2_)
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def crop_gligen(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    if "gligen" not in cond_dict:
        return
    type, model, cond = cond_dict["gligen"]
    if type != "position":
        from warnings import warn
        warn(f"Unknown gligen type {type}")
        return
    cropped = []
    for c in cond:
        emb, h, w, y, x = c
        # Get the coordinates of the box in the upscaled image
        x1 = x * 8
        y1 = y * 8
        x2 = x1 + w * 8
        y2 = y1 + h * 8
        gligen_upscaled_box = resize_region((x1, y1, x2, y2), init_size, canvas_size)

        # Calculate the intersection of the gligen box and the region
        intersection = region_intersection(gligen_upscaled_box, region)
        if intersection is None:
            continue
        x1, y1, x2, y2 = intersection

        # Offset the gligen box so that the origin is at the top left of the tile region
        x1 -= region[0]
        y1 -= region[1]
        x2 -= region[0]
        y2 -= region[1]

        # Add the padding
        x1 += w_pad
        y1 += h_pad
        x2 += w_pad
        y2 += h_pad

        # Set the new position params
        h = (y2 - y1) // 8
        w = (x2 - x1) // 8
        x = x1 // 8
        y = y1 // 8
        cropped.append((emb, h, w, y, x))

    cond_dict["gligen"] = (type, model, cropped)


def crop_area(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    if "area" not in cond_dict:
        return

    # Resize the area conditioning to the canvas size and confine it to the tile region
    h, w, y, x = cond_dict["area"]
    w, h, x, y = 8 * w, 8 * h, 8 * x, 8 * y
    x1, y1, x2, y2 = resize_region((x, y, x + w, y + h), init_size, canvas_size)
    intersection = region_intersection((x1, y1, x2, y2), region)
    if intersection is None:
        del cond_dict["area"]
        del cond_dict["strength"]
        return
    x1, y1, x2, y2 = intersection

    # Offset origin to the top left of the tile
    x1 -= region[0]
    y1 -= region[1]
    x2 -= region[0]
    y2 -= region[1]

    # Add the padding
    x1 += w_pad
    y1 += h_pad
    x2 += w_pad
    y2 += h_pad

    # Set the params for tile
    w, h = (x2 - x1) // 8, (y2 - y1) // 8
    x, y = x1 // 8, y1 // 8

    cond_dict["area"] = (h, w, y, x)


def crop_mask(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    if "mask" not in cond_dict:
        return
    mask_tensor = cond_dict["mask"]  # (B, H, W)
    masks = []
    for i in range(mask_tensor.shape[0]):
        # Convert to PIL image
        mask = tensor_to_pil(mask_tensor, i)  # W x H

        # Resize the mask to the canvas size
        mask = mask.resize(canvas_size, Image.Resampling.BICUBIC)

        # Crop the mask to the region
        mask = mask.crop(region)

        # Add padding
        mask, _ = resize_and_pad_image(mask, tile_size[0], tile_size[1], fill=True)

        # Resize the mask to the tile size
        if tile_size != mask.size:
            mask = mask.resize(tile_size, Image.Resampling.BICUBIC)

        # Convert back to tensor
        mask = pil_to_tensor(mask)  # (1, H, W, 1)
        mask = mask.squeeze(-1)  # (1, H, W)
        masks.append(mask)

    cond_dict["mask"] = torch.cat(masks, dim=0)  # (B, H, W)


def crop_cond(cond, region, init_size, canvas_size, tile_size, w_pad=0, h_pad=0):
    cropped = []
    for emb, x in cond:
        cond_dict = x.copy()
        n = [emb, cond_dict]
        crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_gligen(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_area(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_mask(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        cropped.append(n)
    return cropped


from PIL import Image

from workflow_api import ImageUpscaleWithModel

if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image


class Upscaler:

    def _upscale(self, img: Image, scale):
        global actual_upscaler
        if (actual_upscaler is None):
            return img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
        tensor = pil_to_tensor(img)
        image_upscale_node = ImageUpscaleWithModel()
        (upscaled,) = image_upscale_node.upscale(actual_upscaler, tensor)
        return tensor_to_pil(upscaled)

    def upscale(self, img: Image, scale, selected_model: str = None):
        global batch
        batch = [self._upscale(img, scale) for img in batch]
        return batch[0]


class UpscalerData:
    name = ""
    data_path = ""

    def __init__(self):
        self.scaler = Upscaler()


import math

import torch
from PIL import Image, ImageFilter

from nodes import common_ksampler, VAEEncode, VAEDecode

if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image


class StableDiffusionProcessing:

    def __init__(self, init_img, model, positive, negative, vae, seed, steps, cfg, sampler_name, scheduler, denoise,
                 upscale_by, uniform_tile_mode):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = None
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width
        self.height = init_img.height

        # ComfyUI Sampler inputs
        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode

        # Other required A1111 variables for the USDU script that is currently unused in this script
        self.extra_generation_params = {}


class Processed:

    def __init__(self, p: StableDiffusionProcessing, images: list, seed: int, info: str):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        return None


def fix_seed(p: StableDiffusionProcessing):
    pass


def process_images(p: StableDiffusionProcessing) -> Processed:
    # Where the main image generation happens in A1111

    # Setup
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    if p.uniform_tile_mode == "enable":
        # Expand the crop region to match the processing size ratio and then resize it to the processing size
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_ratio = crop_width / crop_height
        p_ratio = p.width / p.height
        if crop_ratio > p_ratio:
            target_width = crop_width
            target_height = round(crop_width / p_ratio)
        else:
            target_width = round(crop_height * p_ratio)
            target_height = crop_height
        crop_region, _ = expand_crop(crop_region, image_mask.width, image_mask.height, target_width, target_height)
        tile_size = p.width, p.height
    else:
        # Uses the minimal size that can fit the mask, minimizes tile size but may lead to image sizes that the model is not trained on
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        target_width = math.ceil(crop_width / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        crop_region, tile_size = expand_crop(crop_region, image_mask.width,
                                             image_mask.height, target_width, target_height)

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    global batch
    tiles = [img.crop(crop_region) for img in batch]

    # Assume the same size for all images in the batch
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
    negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

    # Encode the image
    vae_encoder = VAEEncode()
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = vae_encoder.encode(p.vae, batched_tiles)

    # Generate samples
    (samples,) = common_ksampler(p.model, p.seed, p.steps, p.cfg, p.sampler_name,
                                 p.scheduler, positive_cropped, negative_cropped, latent, denoise=p.denoise)

    # Decode the sample
    vae_decoder = VAEDecode()
    (decoded,) = vae_decoder.decode(p.vae, samples)

    # Convert the sample to a PIL image
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        # Put the tile into position
        image_tile_only = Image.new('RGBA', init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        image_mask = image_mask.resize(temp.size) # FIXME: This is a workaround for a bug in the original code
        temp.putalpha(image_mask)
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert('RGBA')
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert('RGB')
        batch[i] = result

    processed = Processed(p, [batch[0]], p.seed, None)
    return processed


import math
from enum import Enum

from PIL import Image, ImageDraw, ImageOps




class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2


class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


class USDUpscaler():

    def __init__(self, p, image, upscaler_index: int, save_redraw, save_seams_fix, tile_width, tile_height) -> None:
        self.p: StableDiffusionProcessing = p
        self.image: Image = image
        self.scale_factor = math.ceil(max(p.width, p.height) / max(image.width, image.height))
        global sd_upscalers
        self.upscaler = sd_upscalers[upscaler_index]
        self.redraw = USDURedraw()
        self.redraw.save = save_redraw
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.save = save_seams_fix
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width
        self.initial_info = None
        self.rows = math.ceil(self.p.height / self.redraw.tile_height)
        self.cols = math.ceil(self.p.width / self.redraw.tile_width)

    def get_factor(self, num):
        # Its just return, don't need elif
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self):
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale_factor == 0:
            self.scale_factor += 1
            current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
            if current_scale_factor == 0:
                break
        self.scales = enumerate(scales)

    def upscale(self):
        # Log info
        print(f"Canva size: {self.p.width}x{self.p.height}")
        print(f"Image size: {self.image.width}x{self.image.height}")
        print(f"Scale factor: {self.scale_factor}")
        # Check upscaler is not empty
        if self.upscaler.name == "None":
            self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
            return
        # Get list with scale factors
        self.get_factors()
        # Upscaling image over all factors
        for index, value in self.scales:
            print(f"Upscaling iteration {index + 1} with scale factor {value}")
            self.image = self.upscaler.scaler.upscale(self.image, value, self.upscaler.data_path)
        # Resize image to set values
        self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)

    def setup_redraw(self, redraw_mode, padding, mask_blur):
        self.redraw.mode = USDUMode(redraw_mode)
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding
        self.p.mask_blur = mask_blur

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def save_image(self):
        if type(self.p.prompt) != list:
            save_image(self.image, self.p.outpath_samples, "", self.p.seed, self.p.prompt, opts.samples_format,
                              info=self.initial_info, p=self.p)
        else:
            save_image(self.image, self.p.outpath_samples, "", self.p.seed, self.p.prompt[0],
                              opts.samples_format, info=self.initial_info, p=self.p)

    def calc_jobs_count(self):
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = 0
        if self.seams_fix.mode == USDUSFMode.BAND_PASS:
            seams_job_count = self.rows + self.cols - 2
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols + (self.rows - 1) * (
                        self.cols - 1)
        global state
        state.job_count = redraw_job_count + seams_job_count

    def print_info(self):
        print(f"Tile size: {self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles amount: {self.rows * self.cols}")
        print(f"Grid: {self.rows}x{self.cols}")
        print(f"Redraw enabled: {self.redraw.enabled}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")

    def add_extra_info(self):
        self.p.extra_generation_params["Ultimate SD upscale upscaler"] = self.upscaler.name
        self.p.extra_generation_params["Ultimate SD upscale tile_width"] = self.redraw.tile_width
        self.p.extra_generation_params["Ultimate SD upscale tile_height"] = self.redraw.tile_height
        self.p.extra_generation_params["Ultimate SD upscale mask_blur"] = self.p.mask_blur
        self.p.extra_generation_params["Ultimate SD upscale padding"] = self.redraw.padding

    def process(self):
        global state
        state.begin()
        self.calc_jobs_count()
        self.result_images = []
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)
        if self.redraw.save:
            self.save_image()

        if self.seams_fix.enabled:
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
            if self.seams_fix.save:
                self.save_image()
        state.end()


class USDURedraw():

    def init_draw(self, p, width, height):
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi, yi):
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height

        return x1, y1, x2, y2

    def linear_process(self, p, image, rows, cols):
        global state
        mask, draw = self.init_draw(p, image.width, image.height)
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def chess_process(self, p, image, rows, cols):
        global state
        mask, draw = self.init_draw(p, image.width, image.height)
        tiles = []
        # calc tiles colors
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                if xi == 0:
                    tiles.append([])
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                tiles[yi].append(color)

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    tiles[yi][xi] = not tiles[yi][xi]
                    continue
                tiles[yi][xi] = not tiles[yi][xi]
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    continue
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        self.initial_info = None
        if self.mode == USDUMode.LINEAR:
            return self.linear_process(p, image, rows, cols)
        if self.mode == USDUMode.CHESS:
            return self.chess_process(p, image, rows, cols)


class USDUSeamsFix():

    def init_draw(self, p):
        self.initial_info = None
        p.width = math.ceil((self.tile_width + self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height + self.padding) / 64) * 64

    def half_tile_process(self, p, image, rows, cols):
        global state
        self.init_draw(p)
        processed = None

        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(gradient.resize(
            (self.tile_width, self.tile_height // 2), resample=Image.BICUBIC), (0, 0))
        row_gradient.paste(gradient.rotate(180).resize(
            (self.tile_width, self.tile_height // 2), resample=Image.BICUBIC),
            (0, self.tile_height // 2))
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(gradient.rotate(90).resize(
            (self.tile_width // 2, self.tile_height), resample=Image.BICUBIC), (0, 0))
        col_gradient.paste(gradient.rotate(270).resize(
            (self.tile_width // 2, self.tile_height), resample=Image.BICUBIC), (self.tile_width // 2, 0))

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        for yi in range(rows - 1):
            for xi in range(cols):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(row_gradient, (xi * self.tile_width, yi * self.tile_height + self.tile_height // 2))

                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        for yi in range(rows):
            for xi in range(cols - 1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(col_gradient, (xi * self.tile_width + self.tile_width // 2, yi * self.tile_height))

                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def half_tile_process_corners(self, p, image, rows, cols):
        global state
        fixed_image = self.half_tile_process(p, image, rows, cols)
        processed = None
        self.init_draw(p)
        gradient = Image.radial_gradient("L").resize(
            (self.tile_width, self.tile_height), resample=Image.BICUBIC)
        gradient = ImageOps.invert(gradient)
        p.denoising_strength = self.denoise
        # p.mask_blur = 0
        p.mask_blur = self.mask_blur

        for yi in range(rows - 1):
            for xi in range(cols - 1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = 0
                mask = Image.new("L", (fixed_image.width, fixed_image.height), "black")
                mask.paste(gradient, (xi * self.tile_width + self.tile_width // 2,
                                      yi * self.tile_height + self.tile_height // 2))

                p.init_images = [fixed_image]
                p.image_mask = mask
                processed = process_images(p)
                if (len(processed.images) > 0):
                    fixed_image = processed.images[0]

        p.width = fixed_image.width
        p.height = fixed_image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return fixed_image

    def band_pass_process(self, p, image, cols, rows):
        global state
        self.init_draw(p)
        processed = None

        p.denoising_strength = self.denoise
        p.mask_blur = 0

        gradient = Image.linear_gradient("L")
        mirror_gradient = Image.new("L", (256, 256), "black")
        mirror_gradient.paste(gradient.resize((256, 128), resample=Image.BICUBIC), (0, 0))
        mirror_gradient.paste(gradient.rotate(180).resize((256, 128), resample=Image.BICUBIC), (0, 128))

        row_gradient = mirror_gradient.resize((image.width, self.width), resample=Image.BICUBIC)
        col_gradient = mirror_gradient.rotate(90).resize((self.width, image.height), resample=Image.BICUBIC)

        for xi in range(1, rows):
            if state.interrupted:
                break
            p.width = self.width + self.padding * 2
            p.height = image.height
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(col_gradient, (xi * self.tile_width - self.width // 2, 0))

            p.init_images = [image]
            p.image_mask = mask
            processed = process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]
        for yi in range(1, cols):
            if state.interrupted:
                break
            p.width = image.width
            p.height = self.width + self.padding * 2
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(row_gradient, (0, yi * self.tile_height - self.width // 2))

            p.init_images = [image]
            p.image_mask = mask
            processed = process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        if USDUSFMode(self.mode) == USDUSFMode.BAND_PASS:
            return self.band_pass_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE:
            return self.half_tile_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            return self.half_tile_process_corners(p, image, rows, cols)
        else:
            return image


class Script(Script):
    def title(self):
        return "Ultimate SD upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        target_size_types = [
            "From img2img2 settings",
            "Custom size",
            "Scale from image size"
        ]

        seams_fix_types = [
            "None",
            "Band pass",
            "Half tile offset pass",
            "Half tile offset pass + intersections"
        ]

        redrow_modes = [
            "Linear",
            "Chess",
            "None"
        ]

    def run(self, p, _, tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise,
            seams_fix_padding,
            upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image, seams_fix_mask_blur,
            seams_fix_type, target_size_type, custom_width, custom_height, custom_scale):

        # Init
        fix_seed(p)
        torch_gc()

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.inpaint_full_res = False

        p.inpainting_fill = 1
        p.n_iter = 1
        p.batch_size = 1

        seed = p.seed

        # Init image
        init_img = p.init_images[0]
        if init_img == None:
            return Processed(p, [], seed, "Empty image")
        init_img = flatten(init_img, opts.img2img_background_color)

        # override size
        if target_size_type == 1:
            p.width = custom_width
            p.height = custom_height
        if target_size_type == 2:
            p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
            p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

        # Upscaling
        upscaler = USDUpscaler(p, init_img, upscaler_index, save_upscaled_image, save_seams_fix_image, tile_width,
                               tile_height)
        upscaler.upscale()

        # Drawing
        upscaler.setup_redraw(redraw_mode, padding, mask_blur)
        upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur, seams_fix_width,
                                 seams_fix_type)
        upscaler.print_info()
        upscaler.add_extra_info()
        upscaler.process()
        result_images = upscaler.result_images

        return Processed(p, result_images, seed, upscaler.initial_info if upscaler.initial_info is not None else "")


# Make some patches to the script
import math

from PIL import Image

if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image

#
# Instead of using multiples of 64, use multiples of 8
#

# Upscaler
old_init = USDUpscaler.__init__


def new_init(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height):
    p.width = math.ceil((image.width * p.upscale_by) / 8) * 8
    p.height = math.ceil((image.height * p.upscale_by) / 8) * 8
    old_init(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height)


USDUpscaler.__init__ = new_init

# Redraw
old_setup_redraw = USDURedraw.init_draw


def new_setup_redraw(self, p, width, height):
    mask, draw = old_setup_redraw(self, p, width, height)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8
    return mask, draw


USDURedraw.init_draw = new_setup_redraw

# Seams fix
old_setup_seams_fix = USDUSeamsFix.init_draw


def new_setup_seams_fix(self, p):
    old_setup_seams_fix(self, p)
    p.width = math.ceil((self.tile_width + self.padding) / 8) * 8
    p.height = math.ceil((self.tile_height + self.padding) / 8) * 8


USDUSeamsFix.init_draw = new_setup_seams_fix

#
# Make the script upscale on a batch of images instead of one image
#

old_upscale = USDUpscaler.upscale


def new_upscale(self):
    old_upscale(self)
    global batch
    batch = [self.image] + \
                   [img.resize((self.p.width, self.p.height), resample=Image.LANCZOS) for img in batch[1:]]


USDUpscaler.upscale = new_upscale


# ComfyUI Node for Ultimate SD Upscale by Coyote-A: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111

import torch

import comfy

MAX_RESOLUTION = 8192
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": USDUMode.LINEAR,
    "Chess": USDUMode.CHESS,
    "None": USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": USDUSFMode.NONE,
    "Band Pass": USDUSFMode.BAND_PASS,
    "Half Tile": USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


def USDU_base_inputs():
    return [
        ("image", ("IMAGE",)),
        # Sampling Params
        ("model", ("MODEL",)),
        ("positive", ("CONDITIONING",)),
        ("negative", ("CONDITIONING",)),
        ("vae", ("VAE",)),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS,)),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS,)),
        ("denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL",)),
        ("mode_type", (list(MODES.keys()),)),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()),)),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8})),
        # Misc
        ("force_uniform_tiles", (["enable", "disable"],))
    ]


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type in required:
            inputs["required"][name] = type
    if optional:
        inputs["optional"] = {}
        for name, type in optional:
            inputs["optional"][name] = type
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break


class UltimateSDUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return prepare_inputs(USDU_base_inputs())

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles):
        #
        # Set up A1111 patches
        #

        # Upscaler
        # An object that the script works with
        global sd_upscalers, actual_upscaler, batch
        sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        actual_upscaler = upscale_model

        # Set the batch of images
        batch = [tensor_to_pil(image, i) for i in range(len(image))]

        # Processing
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(image), model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by, force_uniform_tiles
        )

        #
        # Running the script
        #
        script = Script()
        script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                   mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                   seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                   upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[mode_type],
                   save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                   seams_fix_type=SEAM_FIX_MODES[seam_fix_mode], target_size_type=2,
                   custom_width=None, custom_height=None, custom_scale=upscale_by)

        # Return the resulting images
        images = [pil_to_tensor(img) for img in batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)


class UltimateSDUpscaleNoUpscale:
    @classmethod
    def INPUT_TYPES(s):
        required = USDU_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        rename_input(required, "image", "upscaled_image")
        return prepare_inputs(required)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, upscaled_image, model, positive, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles):
        global sd_upscalers, actual_upscaler, batch
        sd_upscalers[0] = UpscalerData()
        actual_upscaler = None
        batch = [tensor_to_pil(upscaled_image, i) for i in range(len(upscaled_image))]
        sdprocessing = StableDiffusionProcessing(
            tensor_to_pil(upscaled_image), model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, 1, force_uniform_tiles
        )

        script = Script()
        processed = script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                               mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                               seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=MODES[mode_type],
                               save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                               seams_fix_type=SEAM_FIX_MODES[seam_fix_mode], target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=1)

        images = [pil_to_tensor(img) for img in batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscale": UltimateSDUpscale,
    "UltimateSDUpscaleNoUpscale": UltimateSDUpscaleNoUpscale
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscale": "Ultimate SD Upscale",
    "UltimateSDUpscaleNoUpscale": "Ultimate SD Upscale (No Upscale)"
}
