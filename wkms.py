from collections import namedtuple
import configparser
import math
import os
import random
import re
import threading
import numpy as np
import torch
import workflow_api as ld
from segment_anything import SamPredictor, sam_model_registry

def try_install_custom_node(custom_node_url, msg):
    try:
        import cm_global
        cm_global.try_call(api='cm.try-install-custom-node',
                           sender="Impact Pack", custom_node_url=custom_node_url, msg=msg)
    except Exception:
        print(msg)
        print(f"[Impact Pack] ComfyUI-Manager is outdated. The custom node installation feature is not available.")

def sam_predict(predictor, points, plabs, bbox, threshold):
    point_coords = None if not points else np.array(points)
    point_labels = None if not plabs else np.array(plabs)

    box = np.array([bbox]) if bbox is not None else None

    cur_masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box)

    total_masks = []

    selected = False
    max_score = 0
    max_mask = None
    for idx in range(len(scores)):
        if scores[idx] > max_score:
            max_score = scores[idx]
            max_mask = cur_masks[idx]

        if scores[idx] >= threshold:
            selected = True
            total_masks.append(cur_masks[idx])
        else:
            pass

    if not selected and max_mask is not None:
        total_masks.append(max_mask)

    return total_masks

def is_same_device(a, b):
    a_device = torch.device(a) if isinstance(a, str) else a
    b_device = torch.device(b) if isinstance(b, str) else b
    return a_device.type == b_device.type and a_device.index == b_device.index

class SafeToGPU:
    def __init__(self, size):
        self.size = size

    def to_device(self, obj, device):
        if is_same_device(device, 'cpu'):
            obj.to(device)
        else:
            if is_same_device(obj.device, 'cpu'):  # cpu to gpu
                ld.free_memory(self.size * 1.3, device)
                if ld.get_free_memory(device) > self.size * 1.3:
                    try:
                        obj.to(device)
                    except:
                        print(f"WARN: The model is not moved to the '{device}' due to insufficient memory. [1]")
                else:
                    print(f"WARN: The model is not moved to the '{device}' due to insufficient memory. [2]")


class SAMWrapper:
    def __init__(self, model, is_auto_mode, safe_to_gpu=None):
        self.model = model
        self.safe_to_gpu = safe_to_gpu if safe_to_gpu is not None else SafeToGPU_stub()
        self.is_auto_mode = is_auto_mode

    def prepare_device(self):
        if self.is_auto_mode:
            device = ld.get_torch_device()
            self.safe_to_gpu.to_device(self.model, device=device)

    def release_device(self):
        if self.is_auto_mode:
            self.model.to(device="cpu")

    def predict(self, image, points, plabs, bbox, threshold):
        predictor = SamPredictor(self.model)
        predictor.set_image(image, "RGB")

        return sam_predict(predictor, points, plabs, bbox, threshold)


class ESAMWrapper:
    def __init__(self, model, device):
        self.model = model
        self.func_inference = ld.NODE_CLASS_MAPPINGS['Yoloworld_ESAM_Zho']
        self.device = device

    def prepare_device(self):
        pass

    def release_device(self):
        pass

    def predict(self, image, points, plabs, bbox, threshold):
        if self.device == 'CPU':
            self.device = 'cpu'
        else:
            self.device = 'cuda'

        detected_masks = self.func_inference.inference_sam_with_boxes(image=image, xyxy=[bbox], model=self.model, device=self.device)
        return [detected_masks.squeeze(0)]

class SAMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        models = [x for x in ld.get_filename_list("sams") if 'hq' not in x]
        return {
            "required": {
                "model_name": (models + ['ESAM'], {"tooltip": "The detection accuracy varies depending on the SAM model. ESAM can only be used if ComfyUI-YoloWorld-EfficientSAM is installed."}),
                "device_mode": (["AUTO", "Prefer GPU", "CPU"], {"tooltip": "AUTO: Only applicable when a GPU is available. It temporarily loads the SAM_MODEL into VRAM only when the detection function is used.\n"
                                                                           "Prefer GPU: Tries to keep the SAM_MODEL on the GPU whenever possible. This can be used when there is sufficient VRAM available.\n"
                                                                           "CPU: Always loads only on the CPU."}),
            }
        }

    RETURN_TYPES = ("SAM_MODEL", )
    FUNCTION = "load_model"

    CATEGORY = "ImpactPack"

    DESCRIPTION = "Load the SAM (Segment Anything) model. This can be used in places that utilize SAM detection functionality, such as SAMDetector or SimpleDetector.\nThe SAM detection functionality in Impact Pack must use the SAM_MODEL loaded through this node."

    def load_model(self, model_name, device_mode="auto"):
        if model_name == 'ESAM':
            if 'ESAM_ModelLoader_Zho' not in ld.NODE_CLASS_MAPPINGS:
                try_install_custom_node('https://github.com/ZHO-ZHO-ZHO/ComfyUI-YoloWorld-EfficientSAM',
                                        "To use 'ESAM' model, 'ComfyUI-YoloWorld-EfficientSAM' extension is required.")
                raise Exception("'ComfyUI-YoloWorld-EfficientSAM' node isn't installed.")

            esam_loader = ld.NODE_CLASS_MAPPINGS['ESAM_ModelLoader_Zho']()

            if device_mode == 'CPU':
                esam = esam_loader.load_esam_model('CPU')[0]
            else:
                device_mode = 'CUDA'
                esam = esam_loader.load_esam_model('CUDA')[0]

            sam_obj = ESAMWrapper(esam, device_mode)
            esam.sam_wrapper = sam_obj
            
            print(f"Loads EfficientSAM model: (device:{device_mode})")
            return (esam, )

        modelname = 'E:\\Dev\\LightDiffusion\\_internal\\sam_vit_b_01ec64.pth'

        if 'vit_h' in model_name:
            model_kind = 'vit_h'
        elif 'vit_l' in model_name:
            model_kind = 'vit_l'
        else:
            model_kind = 'vit_b'

        sam = sam_model_registry[model_kind](checkpoint=modelname)
        size = os.path.getsize(modelname)
        safe_to = SafeToGPU(size)

        # Unless user explicitly wants to use CPU, we use GPU
        device = ld.get_torch_device() if device_mode == "Prefer GPU" else "CPU"

        if device_mode == "Prefer GPU":
            safe_to.to_device(sam, device)

        is_auto_mode = device_mode == "AUTO"

        sam_obj = SAMWrapper(sam, is_auto_mode=is_auto_mode, safe_to_gpu=safe_to)
        sam.sam_wrapper = sam_obj

        print(f"Loads SAM model: {modelname} (device:{device_mode})")
        return (sam, )

from PIL import Image
import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image
import torch

orig_torch_load = torch.load

try:
    from ultralytics import YOLO
except Exception as e:
    print(e)
    print(f"\n!!!!!\n\n[ComfyUI-Impact-Subpack] If this error occurs, please check the following link:\n\thttps://github.com/ltdrdata/ComfyUI-Impact-Pack/blob/Main/troubleshooting/TROUBLESHOOTING.md\n\n!!!!!\n")
    raise e

# HOTFIX: https://github.com/ltdrdata/ComfyUI-Impact-Pack/issues/754
# importing YOLO breaking original torch.load capabilities
torch.load = orig_torch_load  

def load_yolo(model_path: str):
    try:
        return YOLO(model_path)
    except ModuleNotFoundError:
        # https://github.com/ultralytics/ultralytics/issues/3856
        YOLO("yolov8n.pt")
        return YOLO(model_path)


def inference_bbox(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
):
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    cv2_image = np.array(image)
    if len(cv2_image.shape) == 3:
        cv2_image = cv2_image[:, :, ::-1].copy()  # Convert RGB to BGR for cv2 processing
    else:
        # Handle the grayscale image here
        # For example, you might want to convert it to a 3-channel grayscale image for consistency:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results


def inference_segm(
    model,
    image: Image.Image,
    confidence: float = 0.3,
    device: str = "",
):
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    # NOTE: masks.data will be None when n == 0
    segms = pred[0].masks.data.cpu().numpy()

    h_segms = segms.shape[1]
    w_segms = segms.shape[2]
    h_orig = image.size[1]
    w_orig = image.size[0]
    ratio_segms = h_segms / w_segms
    ratio_orig = h_orig / w_orig

    if ratio_segms == ratio_orig:
        h_gap = 0
        w_gap = 0
    elif ratio_segms > ratio_orig:
        h_gap = int((ratio_segms - ratio_orig) * h_segms)
        w_gap = 0
    else:
        h_gap = 0
        ratio_segms = w_segms / h_segms
        ratio_orig = w_orig / h_orig
        w_gap = int((ratio_segms - ratio_orig) * w_segms)

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])

        mask = torch.from_numpy(segms[i])
        mask = mask[h_gap:mask.shape[0] - h_gap, w_gap:mask.shape[1] - w_gap]

        scaled_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(image.size[1], image.size[0]),
                                                      mode='bilinear', align_corners=False)
        scaled_mask = scaled_mask.squeeze().squeeze()

        results[2].append(scaled_mask.numpy())
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results

def _tensor_check_image(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")
    return


def tensor2pil(image):
    _tensor_check_image(image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8))

def create_segmasks(results):
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results

my_path = os.path.dirname(__file__)
old_config_path = os.path.join(my_path, "impact-pack.ini")
config_path = os.path.join(my_path, "..", "..", "impact-pack.ini")
latent_letter_path = os.path.join(my_path, "..", "..", "latent.png")

def read_config():
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        default_conf = config['default']

        if not os.path.exists(default_conf['custom_wildcards']):
            print(f"[WARN] ComfyUI-Impact-Pack: custom_wildcards path not found: {default_conf['custom_wildcards']}. Using default path.")
            default_conf['custom_wildcards'] = os.path.join(my_path, "..", "..", "custom_wildcards")

        return {
                    'dependency_version': int(default_conf['dependency_version']),
                    'mmdet_skip': default_conf['mmdet_skip'].lower() == 'true' if 'mmdet_skip' in default_conf else True,
                    'sam_editor_cpu': default_conf['sam_editor_cpu'].lower() == 'true' if 'sam_editor_cpu' in default_conf else False,
                    'sam_editor_model': default_conf['sam_editor_model'].lower() if 'sam_editor_model' else 'sam_vit_b_01ec64.pth',
                    'custom_wildcards': default_conf['custom_wildcards'] if 'custom_wildcards' in default_conf else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "custom_wildcards")),
                    'disable_gpu_opencv': default_conf['disable_gpu_opencv'].lower() == 'true' if 'disable_gpu_opencv' in default_conf else True
               }

    except Exception:
        return {
            'dependency_version': 0,
            'mmdet_skip': True,
            'sam_editor_cpu': False,
            'sam_editor_model': 'sam_vit_b_01ec64.pth',
            'custom_wildcards': os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "custom_wildcards")),
            'disable_gpu_opencv': True
        }


cached_config = None


def get_config():
    global cached_config

    if cached_config is None:
        cached_config = read_config()

    return cached_config

def use_gpu_opencv():
    return not get_config()['disable_gpu_opencv']

def dilate_masks(segmasks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return segmasks

    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    if use_gpu_opencv():
        kernel = cv2.UMat(kernel)

    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]

        if use_gpu_opencv():
            cv2_mask = cv2.UMat(cv2_mask)

        if dilation_factor > 0:
            dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
        else:
            dilated_mask = cv2.erode(cv2_mask, kernel, iter)

        if use_gpu_opencv():
            dilated_mask = dilated_mask.get()

        item = (segmasks[i][0], dilated_mask, segmasks[i][2])
        dilated_masks.append(item)

    return dilated_masks

def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp+size)

    return int(new_startp), int(new_endp)


def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]

def crop_ndarray2(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[y1:y2, x1:x2]

    return cropped

def crop_ndarray4(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped


crop_tensor4 = crop_ndarray4

def crop_image(image, crop_region):
    return crop_tensor4(image, crop_region)

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

def combine_masks(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0][1])
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i][1])

            if combined_cv2_mask.shape == cv2_mask.shape:
                combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
            else:
                # do nothing - incompatible mask
                pass

        mask = torch.from_numpy(combined_cv2_mask)
        return mask

class UltraBBoxDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        detected_results = inference_bbox(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)

                items.append(item)

        shape = image.shape[1], image.shape[2]
        segs = shape, items

        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)

        return segs

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_bbox(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        return combine_masks(segmasks)

    def setAux(self, x):
        pass


class UltraSegmDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        detected_results = inference_segm(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)

                items.append(item)

        shape = image.shape[1], image.shape[2]
        segs = shape, items

        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)

        return segs

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_segm(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        return combine_masks(segmasks)

    def setAux(self, x):
        pass

def add_folder_path_and_extensions(folder_name, full_folder_paths, extensions):
    # Iterate over the list of full folder paths
    for full_folder_path in full_folder_paths:
        # Use the provided function to add each model folder path
        ld.add_model_folder_path(folder_name, full_folder_path)

    # Now handle the extensions. If the folder name already exists, update the extensions
    if folder_name in ld.folder_names_and_paths:
        # Unpack the current paths and extensions
        current_paths, current_extensions = ld.folder_names_and_paths[folder_name]
        # Update the extensions set with the new extensions
        updated_extensions = current_extensions | extensions
        # Reassign the updated tuple back to the dictionary
        ld.folder_names_and_paths[folder_name] = (current_paths, updated_extensions)
    else:
        # If the folder name was not present, add_model_folder_path would have added it with the last path
        # Now we just need to update the set of extensions as it would be an empty set
        # Also ensure that all paths are included (since add_model_folder_path adds only one path at a time)
        ld.folder_names_and_paths[folder_name] = (full_folder_paths, extensions)

model_path = ld.models_dir
add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(model_path, "ultralytics", "bbox")], ld.supported_pt_extensions)
add_folder_path_and_extensions("ultralytics_segm", [os.path.join(model_path, "ultralytics", "segm")], ld.supported_pt_extensions)
add_folder_path_and_extensions("ultralytics", [os.path.join(model_path, "ultralytics")], ld.supported_pt_extensions)

class NO_SEGM_DETECTOR:
    pass

detection_labels = [
    'hand', 'face', 'mouth', 'eyes', 'eyebrows', 'pupils',
    'left_eyebrow', 'left_eye', 'left_pupil', 'right_eyebrow', 'right_eye', 'right_pupil',
    'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear',
    'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress',
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
 ]

class UltralyticsDetectorProvider:
    @classmethod
    def INPUT_TYPES(s):
        bboxs = ["bbox/"+x for x in ld.get_filename_list("ultralytics_bbox")]
        segms = ["segm/"+x for x in ld.get_filename_list("ultralytics_segm")]
        return {"required": {"model_name": (bboxs + segms, )}}
    RETURN_TYPES = ("BBOX_DETECTOR", "SEGM_DETECTOR")
    FUNCTION = "doit"

    CATEGORY = "ImpactPack"

    def doit(self, model_name):
        model_path = "E:\\Dev\\LightDiffusion\\_internal\\face_yolov8m.pt"
        model = load_yolo(model_path)

        if model_name.startswith("bbox"):
            return UltraBBoxDetector(model), NO_SEGM_DETECTOR()
        else:
            return UltraBBoxDetector(model), UltraSegmDetector(model)

class SEGSLabelFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS", ),
                        "preset": (['all'] + detection_labels, ),
                        "labels": ("STRING", {"multiline": True, "placeholder": "List the types of segments to be allowed, separated by commas"}),
                     },
                }

    RETURN_TYPES = ("SEGS", "SEGS",)
    RETURN_NAMES = ("filtered_SEGS", "remained_SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def filter(segs, labels):
        labels = set([label.strip() for label in labels])

        if 'all' in labels:
            return (segs, (segs[0], []), )
        else:
            res_segs = []
            remained_segs = []

            for x in segs[1]:
                if x.label in labels:
                    res_segs.append(x)
                elif 'eyes' in labels and x.label in ['left_eye', 'right_eye']:
                    res_segs.append(x)
                elif 'eyebrows' in labels and x.label in ['left_eyebrow', 'right_eyebrow']:
                    res_segs.append(x)
                elif 'pupils' in labels and x.label in ['left_pupil', 'right_pupil']:
                    res_segs.append(x)
                else:
                    remained_segs.append(x)

        return ((segs[0], res_segs), (segs[0], remained_segs), )

    def doit(self, segs, preset, labels):
        labels = labels.split(',')
        return SEGSLabelFilter.filter(segs, labels)

class BboxDetectorForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "bbox_detector": ("BBOX_DETECTOR", ),
                        "image": ("IMAGE", ),
                        "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                        "drop_size": ("INT", {"min": 1, "max": ld.MAX_RESOLUTION, "step": 1, "default": 10}),
                        "labels": ("STRING", {"multiline": True, "default": "all", "placeholder": "List the types of segments to be allowed, separated by commas"}),
                      },
                "optional": {"detailer_hook": ("DETAILER_HOOK",), }
                }

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, bbox_detector, image, threshold, dilation, crop_factor, drop_size, labels=None, detailer_hook=None):
        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: BboxDetectorForEach does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        segs = bbox_detector.detect(image, threshold, dilation, crop_factor, drop_size, detailer_hook)

        if labels is not None and labels != '':
            labels = labels.split(',')
            if len(labels) > 0:
                segs, _ = SEGSLabelFilter.filter(segs, labels)

        return (segs, )

def center_of_bbox(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bbox[0] + w/2, bbox[1] + h/2

def make_2d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)

    elif len(mask.shape) == 3:
        return mask.squeeze(0)

    return mask

def gen_detection_hints_from_mask_area(x, y, mask, threshold, use_negative):
    mask = make_2d_mask(mask)

    points = []
    plabs = []

    # minimum sampling step >= 3
    y_step = max(3, int(mask.shape[0] / 20))
    x_step = max(3, int(mask.shape[1] / 20))

    for i in range(0, len(mask), y_step):
        for j in range(0, len(mask[i]), x_step):
            if mask[i][j] > threshold:
                points.append((x + j, y + i))
                plabs.append(1)
            elif use_negative and mask[i][j] == 0:
                points.append((x + j, y + i))
                plabs.append(0)

    return points, plabs

def gen_negative_hints(w, h, x1, y1, x2, y2):
    npoints = []
    nplabs = []

    # minimum sampling step >= 3
    y_step = max(3, int(w / 20))
    x_step = max(3, int(h / 20))

    for i in range(10, h - 10, y_step):
        for j in range(10, w - 10, x_step):
            if not (x1 - 10 <= j and j <= x2 + 10 and y1 - 10 <= i and i <= y2 + 10):
                npoints.append((j, i))
                nplabs.append(0)

    return npoints, nplabs

def combine_masks2(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0]).astype(np.uint8)
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i]).astype(np.uint8)

            if combined_cv2_mask.shape == cv2_mask.shape:
                combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
            else:
                # do nothing - incompatible mask
                pass

        mask = torch.from_numpy(combined_cv2_mask)
        return mask


def dilate_mask(mask, dilation_factor, iter=1):
    if dilation_factor == 0:
        return make_2d_mask(mask)

    mask = make_2d_mask(mask)

    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    if use_gpu_opencv():
        mask = cv2.UMat(mask)
        kernel = cv2.UMat(kernel)

    if dilation_factor > 0:
        result = cv2.dilate(mask, kernel, iter)
    else:
        result = cv2.erode(mask, kernel, iter)

    if use_gpu_opencv():
        return result.get()
    else:
        return result

def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)

    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)

    return mask

def make_sam_mask(sam, segs, image, detection_hint, dilation,
                  threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):

    if not hasattr(sam, 'sam_wrapper'):
        raise Exception("[Impact Pack] Invalid SAMLoader is connected. Make sure 'SAMLoader (Impact)'.\nKnown issue: The ComfyUI-YOLO node overrides the SAMLoader (Impact), making it unusable. You need to uninstall ComfyUI-YOLO.\n\n\n")

    sam_obj = sam.sam_wrapper
    sam_obj.prepare_device()

    try:
        image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

        total_masks = []

        use_small_negative = mask_hint_use_negative == "Small"

        # seg_shape = segs[0]
        segs = segs[1]
        if detection_hint == "mask-points":
            points = []
            plabs = []

            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = center_of_bbox(segs[i].bbox)
                points.append(center)

                # small point is background, big point is foreground
                if use_small_negative and bbox[2] - bbox[0] < 10:
                    plabs.append(0)
                else:
                    plabs.append(1)

            detected_masks = sam_obj.predict(image, points, plabs, None, threshold)
            total_masks += detected_masks

        else:
            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = center_of_bbox(bbox)

                x1 = max(bbox[0] - bbox_expansion, 0)
                y1 = max(bbox[1] - bbox_expansion, 0)
                x2 = min(bbox[2] + bbox_expansion, image.shape[1])
                y2 = min(bbox[3] + bbox_expansion, image.shape[0])

                dilated_bbox = [x1, y1, x2, y2]

                points = []
                plabs = []
                if detection_hint == "center-1":
                    points.append(center)
                    plabs = [1]  # 1 = foreground point, 0 = background point

                elif detection_hint == "horizontal-2":
                    gap = (x2 - x1) / 3
                    points.append((x1 + gap, center[1]))
                    points.append((x1 + gap * 2, center[1]))
                    plabs = [1, 1]

                elif detection_hint == "vertical-2":
                    gap = (y2 - y1) / 3
                    points.append((center[0], y1 + gap))
                    points.append((center[0], y1 + gap * 2))
                    plabs = [1, 1]

                elif detection_hint == "rect-4":
                    x_gap = (x2 - x1) / 3
                    y_gap = (y2 - y1) / 3
                    points.append((x1 + x_gap, center[1]))
                    points.append((x1 + x_gap * 2, center[1]))
                    points.append((center[0], y1 + y_gap))
                    points.append((center[0], y1 + y_gap * 2))
                    plabs = [1, 1, 1, 1]

                elif detection_hint == "diamond-4":
                    x_gap = (x2 - x1) / 3
                    y_gap = (y2 - y1) / 3
                    points.append((x1 + x_gap, y1 + y_gap))
                    points.append((x1 + x_gap * 2, y1 + y_gap))
                    points.append((x1 + x_gap, y1 + y_gap * 2))
                    points.append((x1 + x_gap * 2, y1 + y_gap * 2))
                    plabs = [1, 1, 1, 1]

                elif detection_hint == "mask-point-bbox":
                    center = center_of_bbox(segs[i].bbox)
                    points.append(center)
                    plabs = [1]

                elif detection_hint == "mask-area":
                    points, plabs = gen_detection_hints_from_mask_area(segs[i].crop_region[0], segs[i].crop_region[1],
                                                                       segs[i].cropped_mask,
                                                                       mask_hint_threshold, use_small_negative)

                if mask_hint_use_negative == "Outter":
                    npoints, nplabs = gen_negative_hints(image.shape[0], image.shape[1],
                                                         segs[i].crop_region[0], segs[i].crop_region[1],
                                                         segs[i].crop_region[2], segs[i].crop_region[3])

                    points += npoints
                    plabs += nplabs

                detected_masks = sam_obj.predict(image, points, plabs, dilated_bbox, threshold)
                total_masks += detected_masks

        # merge every collected masks
        mask = combine_masks2(total_masks)

    finally:
        sam_obj.release_device()

    if mask is not None:
        mask = mask.float()
        mask = dilate_mask(mask.cpu().numpy(), dilation)
        mask = torch.from_numpy(mask)
    else:
        size = image.shape[0], image.shape[1]
        mask = torch.zeros(size, dtype=torch.float32, device="cpu")  # empty mask

    mask = make_3d_mask(mask)
    return mask


SAM_MODEL_TOOLTIP = {"tooltip": "Segment Anything Model for Silhouette Detection.\nBe sure to use the SAM_MODEL loaded through the SAMLoader (Impact) node as input."}
SAM_MODEL_TOOLTIP_OPTIONAL = {"tooltip": "[OPTIONAL]\nSegment Anything Model for Silhouette Detection.\nBe sure to use the SAM_MODEL loaded through the SAMLoader (Impact) node as input.\nGiven this input, it refines the rectangular areas detected by BBOX_DETECTOR into silhouette shapes through SAM.\nsam_model_opt takes priority over segm_detector_opt."}

MASK_HINT_THRESHOLD_TOOLTIP = "When detection_hint is mask-area, the mask of SEGS is used as a point hint for SAM (Segment Anything).\nIn this case, only the areas of the mask with brightness values equal to or greater than mask_hint_threshold are used as hints."
MASK_HINT_USE_NEGATIVE_TOOLTIP = "When detecting with SAM (Segment Anything), negative hints are applied as follows:\nSmall: When the SEGS is smaller than 10 pixels in size\nOuter: Sampling the image area outside the SEGS region at regular intervals"

DILATION_TOOLTIP = "Set the value to dilate the result mask. If the value is negative, it erodes the mask."
DETECTION_HINT_TOOLTIP = {"tooltip": "It is recommended to use only center-1.\nWhen refining the mask of SEGS with the SAM (Segment Anything) model, center-1 uses only the rectangular area of SEGS and a single point at the exact center as hints.\nOther options were added during the experimental stage and do not work well."}

BBOX_EXPANSION_TOOLTIP = "When performing SAM (Segment Anything) detection within the SEGS area, the rectangular area of SEGS is expanded and used as a hint."

class SAMDetectorCombined:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "sam_model": ("SAM_MODEL", SAM_MODEL_TOOLTIP),
                        "segs": ("SEGS", {"tooltip": "This is the segment information detected by the detector.\nIt refines the Mask through the SAM (Segment Anything) detector for all areas pointed to by SEGS, and combines all Masks to return as a single Mask."}),
                        "image": ("IMAGE", {"tooltip": "It is assumed that segs contains only the information about the detected areas, and does not include the image. SAM (Segment Anything) operates by referencing this image."}),
                        "detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area",
                                            "mask-points", "mask-point-bbox", "none"], DETECTION_HINT_TOOLTIP),
                        "dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1, "tooltip": DILATION_TOOLTIP}),
                        "threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Set the sensitivity threshold for the mask detected by SAM (Segment Anything). A higher value generates a more specific mask with a narrower range. For example, when pointing to a person's area, it might detect clothes, which is a narrower range, instead of the entire person."}),
                        "bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "tooltip": BBOX_EXPANSION_TOOLTIP}),
                        "mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": MASK_HINT_THRESHOLD_TOOLTIP}),
                        "mask_hint_use_negative": (["False", "Small", "Outter"], {"tooltip": MASK_HINT_USE_NEGATIVE_TOOLTIP})
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    def doit(self, sam_model, segs, image, detection_hint, dilation,
             threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):
        return (make_sam_mask(sam_model, segs, image, detection_hint, dilation,
                                   threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative), )

def segs_bitwise_and_mask(segs, mask):
    mask = make_2d_mask(mask)

    if mask is None:
        print("[SegsBitwiseAndMask] Cannot operate: MASK is empty.")
        return ([],)

    items = []

    mask = (mask.cpu().numpy() * 255).astype(np.uint8)

    for seg in segs[1]:
        cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
        crop_region = seg.crop_region

        cropped_mask2 = mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]

        new_mask = np.bitwise_and(cropped_mask.astype(np.uint8), cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0

        item = SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
        items.append(item)

    return segs[0], items

class SegsBitwiseAndMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS",),
                        "mask": ("MASK",),
                    }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, segs, mask):
        return (segs_bitwise_and_mask(segs, mask), )

def general_tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear")
    image = image.permute(0, 2, 3, 1)
    return image

class TensorBatchBuilder:
    def __init__(self):
        self.tensor = None

    def concat(self, new_tensor):
        if self.tensor is None:
            self.tensor = new_tensor
        else:
            self.tensor = torch.concat((self.tensor, new_tensor), dim=0)

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
def tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    if image.shape[3] >= 3:
        scaled_images = TensorBatchBuilder()
        for single_image in image:
            single_image = single_image.unsqueeze(0)
            single_pil = tensor2pil(single_image)
            scaled_pil = single_pil.resize((w, h), resample=LANCZOS)

            single_image = pil2tensor(scaled_pil)
            scaled_images.concat(single_image)

        return scaled_images.tensor
    else:
        return general_tensor_resize(image, w, h)

def segs_scale_match(segs, target_shape):
    h = segs[0][0]
    w = segs[0][1]

    th = target_shape[1]
    tw = target_shape[2]

    if (h == th and w == tw) or h == 0 or w == 0:
        return segs

    rh = th / h
    rw = tw / w

    new_segs = []
    for seg in segs[1]:
        cropped_image = seg.cropped_image
        cropped_mask = seg.cropped_mask
        x1, y1, x2, y2 = seg.crop_region
        bx1, by1, bx2, by2 = seg.bbox

        crop_region = int(x1*rw), int(y1*rw), int(x2*rh), int(y2*rh)
        bbox = int(bx1*rw), int(by1*rw), int(bx2*rh), int(by2*rh)
        new_w = crop_region[2] - crop_region[0]
        new_h = crop_region[3] - crop_region[1]

        if isinstance(cropped_mask, np.ndarray):
            cropped_mask = torch.from_numpy(cropped_mask)

        if isinstance(cropped_mask, torch.Tensor) and len(cropped_mask.shape) == 3:
            cropped_mask = torch.nn.functional.interpolate(cropped_mask.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            cropped_mask = cropped_mask.squeeze(0)
        else:
            cropped_mask = torch.nn.functional.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            cropped_mask = cropped_mask.squeeze(0).squeeze(0).numpy()

        if cropped_image is not None:
            cropped_image = tensor_resize(cropped_image if isinstance(cropped_image, torch.Tensor) else torch.from_numpy(cropped_image), new_w, new_h)
            cropped_image = cropped_image.numpy()

        new_seg = SEG(cropped_image, cropped_mask, seg.confidence, crop_region, bbox, seg.label, seg.control_net_wrapper)
        new_segs.append(new_seg)

    return (th, tw), new_segs

def split_string_with_sep(input_string):
    sep_pattern = r'\[SEP(?:\:\w+)?\]'

    substrings = re.split(sep_pattern, input_string)

    result_list = [None]
    matches = re.findall(sep_pattern, input_string)
    for i, substring in enumerate(substrings):
        result_list.append(substring)
        if i < len(matches):
            if matches[i] == '[SEP]':
                result_list.append(None)
            elif matches[i] == '[SEP:R]':
                result_list.append(random.randint(0, 1125899906842624))
            else:
                try:
                    seed = int(matches[i][5:-1])
                except:
                    seed = None
                result_list.append(seed)

    iterable = iter(result_list)
    return list(zip(iterable, iterable))

def starts_with_regex(pattern, text):
    regex = re.compile(pattern)
    return regex.match(text)


def split_to_dict(text):
    pattern = r'\[([A-Za-z0-9_. ]+)\]([^\[]+)(?=\[|$)'
    matches = re.findall(pattern, text)

    result_dict = {key: value.strip() for key, value in matches}

    return result_dict

class WildcardChooser:
    def __init__(self, items, randomize_when_exhaust):
        self.i = 0
        self.items = items
        self.randomize_when_exhaust = randomize_when_exhaust

    def get(self, seg):
        if self.i >= len(self.items):
            self.i = 0
            if self.randomize_when_exhaust:
                random.shuffle(self.items)

        item = self.items[self.i]
        self.i += 1

        return item


class WildcardChooserDict:
    def __init__(self, items):
        self.items = items

    def get(self, seg):
        text = ""
        if 'ALL' in self.items:
            text = self.items['ALL']

        if seg.label in self.items:
            text += self.items[seg.label]

        return text

def process_wildcard_for_segs(wildcard):
    if wildcard.startswith('[LAB]'):
        raw_items = split_to_dict(wildcard)

        items = {}
        for k, v in raw_items.items():
            v = v.strip()
            if v != '':
                items[k] = v

        return 'LAB', WildcardChooserDict(items)

    else:
        match = starts_with_regex(r"\[(ASC-SIZE|DSC-SIZE|ASC|DSC|RND)\]", wildcard)

        if match:
            mode = match[1]
            items = split_string_with_sep(wildcard[len(match[0]):])

            if mode == 'RND':
                random.shuffle(items)
                return mode, WildcardChooser(items, True)
            else:
                return mode, WildcardChooser(items, False)

        else:
            return None, WildcardChooser([(None, wildcard)], False)

class DifferentialDiffusion():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    INIT = False

    def apply(self, model):
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        return (model,)

    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        return (denoise_mask >= threshold).to(denoise_mask.dtype)

def to_tensor(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image)) / 255.0
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, np.ndarray):
        return torch.from_numpy(image)
    raise ValueError(f"Cannot convert {type(image)} to torch.Tensor")

import torchvision

def _tensor_check_mask(mask):
    if mask.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {mask.ndim} dimensions")
    if mask.shape[-1] != 1:
        raise ValueError(f"Expected 1 channel for mask, but found {mask.shape[-1]} channels")
    return

def tensor_gaussian_blur_mask(mask, kernel_size, sigma=10.0):
    """Return NHWC torch.Tenser from ndim == 2 or 4 `np.ndarray` or `torch.Tensor`"""
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if mask.ndim == 2:
        mask = mask[None, ..., None]
    elif mask.ndim == 3:
        mask = mask[..., None]

    _tensor_check_mask(mask)

    if kernel_size <= 0:
        return mask

    kernel_size = kernel_size*2+1

    shortest = min(mask.shape[1], mask.shape[2])
    if shortest <= kernel_size:
        kernel_size = int(shortest/2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            return mask  # skip feathering

    prev_device = mask.device
    device = ld.get_torch_device()
    mask.to(device)

    # apply gaussian blur
    mask = mask[:, None, ..., 0]
    blurred_mask = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(mask)
    blurred_mask = blurred_mask[:, 0, ..., None]

    blurred_mask.to(prev_device)

    return blurred_mask

def crop_ndarray3(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2]

    return cropped

def crop_condition_mask(mask, image, crop_region):
    cond_scale = (mask.shape[1] / image.shape[1], mask.shape[2] / image.shape[2])
    mask_region = [round(v * cond_scale[i % 2]) for i, v in enumerate(crop_region)]
    return crop_ndarray3(mask, mask_region)

def resolve_lora_name(lora_name_cache, name):
    if os.path.exists(name):
        return name
    else:
        if len(lora_name_cache) == 0:
            lora_name_cache.extend(ld.get_filename_list("loras"))

        for x in lora_name_cache:
            if x.endswith(name):
                return x

def process_comment_out(text):
    lines = text.split('\n')

    lines0 = []
    flag = False
    for line in lines:
        if line.lstrip().startswith('#'):
            flag = True
            continue

        if len(lines0) == 0:
            lines0.append(line)
        elif flag:
            lines0[-1] += ' ' + line
            flag = False
        else:
            lines0.append(line)

    return '\n'.join(lines0)

RE_WildCardQuantifier = re.compile(r"(?P<quantifier>\d+)#__(?P<keyword>[\w.\-+/*\\]+?)__", re.IGNORECASE)

wildcard_lock = threading.Lock()

def get_wildcard_dict():
    global wildcard_dict
    with wildcard_lock:
        return wildcard_dict

def is_numeric_string(input_str):
    return re.match(r'^-?\d+(\.\d+)?$', input_str) is not None

def wildcard_normalize(x):
    return x.replace("\\", "/").replace(' ', '-').lower()

def process(text, seed=None):
    text = process_comment_out(text)

    if seed is not None:
        random.seed(seed)
    random_gen = np.random.default_rng(seed)

    local_wildcard_dict = get_wildcard_dict()

    def replace_options(string):
        replacements_found = False

        def replace_option(match):
            nonlocal replacements_found
            options = match.group(1).split('|')

            multi_select_pattern = options[0].split('$$')
            select_range = None
            select_sep = ' '
            range_pattern = r'(\d+)(-(\d+))?'
            range_pattern2 = r'-(\d+)'
            wildcard_pattern = r"__([\w.\-+/*\\]+?)__"

            if len(multi_select_pattern) > 1:
                r = re.match(range_pattern, options[0])

                if r is None:
                    r = re.match(range_pattern2, options[0])
                    a = '1'
                    b = r.group(1).strip()
                else:
                    a = r.group(1).strip()
                    b = r.group(3)
                    if b is not None:
                        b = b.strip()
                        
                if r is not None:
                    if b is not None and is_numeric_string(a) and is_numeric_string(b):
                        # PATTERN: num1-num2
                        select_range = int(a), int(b)
                    elif is_numeric_string(a):
                        # PATTERN: num
                        x = int(a)
                        select_range = (x, x)

                    if select_range is not None and len(multi_select_pattern) == 2:
                        # PATTERN: count$$
                        matches = re.findall(wildcard_pattern, multi_select_pattern[1])
                        if len(options) == 1 and matches:
                            # count$$<single wildcard>
                            options = local_wildcard_dict.get(matches[0])
                        else:
                            # count$$opt1|opt2|...
                            options[0] = multi_select_pattern[1]
                    elif select_range is not None and len(multi_select_pattern) == 3:
                        # PATTERN: count$$ sep $$
                        select_sep = multi_select_pattern[1]
                        options[0] = multi_select_pattern[2]

            adjusted_probabilities = []

            total_prob = 0

            for option in options:
                parts = option.split('::', 1)
                if len(parts) == 2 and is_numeric_string(parts[0].strip()):
                    config_value = float(parts[0].strip())
                else:
                    config_value = 1  # Default value if no configuration is provided

                adjusted_probabilities.append(config_value)
                total_prob += config_value

            normalized_probabilities = [prob / total_prob for prob in adjusted_probabilities]

            if select_range is None:
                select_count = 1
            else:
                select_count = random_gen.integers(low=select_range[0], high=select_range[1]+1, size=1)

            if select_count > len(options):
                random_gen.shuffle(options)
                selected_items = options
            else:
                selected_items = random_gen.choice(options, p=normalized_probabilities, size=select_count, replace=False)

            selected_items2 = [re.sub(r'^\s*[0-9.]+::', '', x, 1) for x in selected_items]
            replacement = select_sep.join(selected_items2)
            if '::' in replacement:
                pass

            replacements_found = True
            return replacement

        pattern = r'{([^{}]*?)}'
        replaced_string = re.sub(pattern, replace_option, string)

        return replaced_string, replacements_found

    def replace_wildcard(string):
        pattern = r"__([\w.\-+/*\\]+?)__"
        matches = re.findall(pattern, string)

        replacements_found = False

        for match in matches:
            keyword = match.lower()
            keyword = wildcard_normalize(keyword)
            if keyword in local_wildcard_dict:
                replacement = random_gen.choice(local_wildcard_dict[keyword])
                replacements_found = True
                string = string.replace(f"__{match}__", replacement, 1)
            elif '*' in keyword:
                subpattern = keyword.replace('*', '.*').replace('+', '\\+')
                total_patterns = []
                found = False
                for k, v in local_wildcard_dict.items():
                    if re.match(subpattern, k) is not None or re.match(subpattern, k+'/') is not None:
                        total_patterns += v
                        found = True

                if found:
                    replacement = random_gen.choice(total_patterns)
                    replacements_found = True
                    string = string.replace(f"__{match}__", replacement, 1)
            elif '/' not in keyword:
                string_fallback = string.replace(f"__{match}__", f"__*/{match}__", 1)
                string, replacements_found = replace_wildcard(string_fallback)

        return string, replacements_found

    replace_depth = 100
    stop_unwrap = False
    while not stop_unwrap and replace_depth > 1:
        replace_depth -= 1  # prevent infinite loop
        
        option_quantifier = [e.groupdict() for e in RE_WildCardQuantifier.finditer(text)]
        for match in option_quantifier:
            keyword = match['keyword'].lower()
            quantifier = int(match['quantifier']) if match['quantifier'] else 1
            replacement = '__|__'.join([keyword,] * quantifier)
            wilder_keyword = keyword.replace('*', '\\*')
            RE_TEMP = re.compile(fr"(?P<quantifier>\d+)#__(?P<keyword>{wilder_keyword})__", re.IGNORECASE)
            text = RE_TEMP.sub(f"__{replacement}__", text)

        # pass1: replace options
        pass1, is_replaced1 = replace_options(text)

        while is_replaced1:
            pass1, is_replaced1 = replace_options(pass1)

        # pass2: replace wildcards
        text, is_replaced2 = replace_wildcard(pass1)
        stop_unwrap = not is_replaced1 and not is_replaced2

    return text

def safe_float(x):
    if is_numeric_string(x):
        return float(x)
    else:
        return 1.0

def extract_lora_values(string):
    pattern = r'<lora:([^>]+)>'
    matches = re.findall(pattern, string)

    def touch_lbw(text):
        return re.sub(r'LBW=[A-Za-z][A-Za-z0-9_-]*:', r'LBW=', text)

    items = [touch_lbw(match.strip(':')) for match in matches]

    added = set()
    result = []
    for item in items:
        item = item.split(':')

        lora = None
        a = None
        b = None
        lbw = None
        lbw_a = None
        lbw_b = None

        if len(item) > 0:
            lora = item[0]

            for sub_item in item[1:]:
                if is_numeric_string(sub_item):
                    if a is None:
                        a = float(sub_item)
                    elif b is None:
                        b = float(sub_item)
                elif sub_item.startswith("LBW="):
                    for lbw_item in sub_item[4:].split(';'):
                        if lbw_item.startswith("A="):
                            lbw_a = safe_float(lbw_item[2:].strip())
                        elif lbw_item.startswith("B="):
                            lbw_b = safe_float(lbw_item[2:].strip())
                        elif lbw_item.strip() != '':
                            lbw = lbw_item

        if a is None:
            a = 1.0
        if b is None:
            b = a

        if lora is not None and lora not in added:
            result.append((lora, a, b, lbw, lbw_a, lbw_b))
            added.add(lora)

    return result

def remove_lora_tags(string):
    pattern = r'<lora:[^>]+>'
    result = re.sub(pattern, '', string)

    return result

def try_install_custom_node(custom_node_url, msg):
    try:
        import cm_global
        cm_global.try_call(api='cm.try-install-custom-node',
                           sender="Impact Pack", custom_node_url=custom_node_url, msg=msg)
    except Exception:
        print(msg)
        print(f"[Impact Pack] ComfyUI-Manager is outdated. The custom node installation feature is not available.")


def process_with_loras(wildcard_opt, model, clip, clip_encoder=None, seed=None, processed=None):
    """
    process wildcard text including loras

    :param wildcard_opt: wildcard text
    :param model: model
    :param clip: clip
    :param clip_encoder: you can pass custom encoder such as adv_cliptext_encode
    :param seed: seed for populating
    :param processed: output variable - [pass1, pass2, pass3] will be saved into passed list
    :return: model, clip, conditioning
    """

    lora_name_cache = []

    pass1 = process(wildcard_opt, seed)
    loras = extract_lora_values(pass1)
    pass2 = remove_lora_tags(pass1)

    for lora_name, model_weight, clip_weight, lbw, lbw_a, lbw_b in loras:
        lora_name_ext = lora_name.split('.')
        if ('.'+lora_name_ext[-1]) not in ld.supported_pt_extensions:
            lora_name = lora_name+".safetensors"

        orig_lora_name = lora_name
        lora_name = resolve_lora_name(lora_name_cache, lora_name)

        if lora_name is not None:
            path = ld.get_full_path("loras", lora_name)
        else:
            path = None

        if path is not None:
            print(f"LOAD LORA: {lora_name}: {model_weight}, {clip_weight}, LBW={lbw}, A={lbw_a}, B={lbw_b}")

            def default_lora():
                return ld.LoraLoader().load_lora(model, clip, lora_name, model_weight, clip_weight)

            if lbw is not None:
                if 'LoraLoaderBlockWeight //Inspire' not in ld.NODE_CLASS_MAPPINGS:
                    ld.try_install_custom_node(
                        'https://github.com/ltdrdata/ComfyUI-Inspire-Pack',
                        "To use 'LBW=' syntax in wildcards, 'Inspire Pack' extension is required.")

                    print(f"'LBW(Lora Block Weight)' is given, but the 'Inspire Pack' is not installed. The LBW= attribute is being ignored.")
                    model, clip = default_lora()
                else:
                    cls = ld.NODE_CLASS_MAPPINGS['LoraLoaderBlockWeight //Inspire']
                    model, clip, _ = cls().doit(model, clip, lora_name, model_weight, clip_weight, False, 0, lbw_a, lbw_b, "", lbw)
            else:
                model, clip = default_lora()
        else:
            print(f"LORA NOT FOUND: {orig_lora_name}")

    pass3 = [x.strip() for x in pass2.split("BREAK")]
    pass3 = [x for x in pass3 if x != '']

    if len(pass3) == 0:
        pass3 = ['']

    pass3_str = [f'[{x}]' for x in pass3]
    print(f"CLIP: {str.join(' + ', pass3_str)}")

    result = None

    for prompt in pass3:
        if clip_encoder is None:
            cur = ld.CLIPTextEncode().encode(clip, prompt)[0]
        else:
            cur = clip_encoder.encode(clip, prompt)[0]

        if result is not None:
            result = ld.ConditioningConcat().concat(result, cur)[0]
        else:
            result = cur

    if processed is not None:
        processed.append(pass1)
        processed.append(pass2)
        processed.append(pass3)

    return model, clip, result


def to_latent_image(pixels, vae):
    x = pixels.shape[1]
    y = pixels.shape[2]
    if pixels.shape[1] != x or pixels.shape[2] != y:
        pixels = pixels[:, :x, :y, :]

    vae_encode = ld.VAEEncode()

    return vae_encode.encode(vae, pixels)[0]

def calculate_sigmas(model, sampler, scheduler, steps):
    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2']:
        steps += 1
        discard_penultimate_sigma = True

    if scheduler.startswith('AYS'):
        sigmas = ld.NODE_CLASS_MAPPINGS['AlignYourStepsScheduler']().get_sigmas(scheduler[4:], steps, denoise=1.0)[0]
    elif scheduler.startswith('GITS[coeff='):
        sigmas = ld.NODE_CLASS_MAPPINGS['GITSScheduler']().get_sigmas(float(scheduler[11:-1]), steps, denoise=1.0)[0]
    else:
        sigmas = ld.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas

def get_noise_sampler(x, cpu, total_sigmas, **kwargs):
    if 'extra_args' in kwargs and 'seed' in kwargs['extra_args']:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        seed = kwargs['extra_args'].get("seed", None)
        return ld.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=cpu)
    return None

def ksampler(sampler_name, total_sigmas, extra_options={}, inpaint_options={}):
    if sampler_name == "dpmpp_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return ld.sample_dpmpp_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return ld.sample_dpmpp_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_2m_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return ld.sample_dpmpp_2m_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_2m_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return ld.sample_dpmpp_2m_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_3m_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return ld.sample_dpmpp_3m_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_3m_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return ld.sample_dpmpp_3m_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    else:
        return ld.sampler_object(sampler_name)

    return ld.KSAMPLER(sampler_function, extra_options, inpaint_options)

class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(latent_image.shape, dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")

class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return ld.prepare_noise(latent_image, self.seed, batch_inds)

class Guider_Basic(ld.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

    return c

def sample_with_custom_noise(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image, noise=None, callback=None):
    latent = latent_image
    latent_image = latent["samples"]

    if hasattr(ld.sample, 'fix_empty_latent_channels'):
        latent_image = ld.fix_empty_latent_channels(model, latent_image)

    out = latent.copy()
    out['samples'] = latent_image

    if noise is None:
        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(out)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(out)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}


    disable_pbar = not ld.PROGRESS_BAR_ENABLED

    device = ld.get_torch_device()

    noise = noise.to(device)
    latent_image = latent_image.to(device)
    if noise_mask is not None:
        noise_mask = noise_mask.to(device)

    if negative != 'NegativePlaceholder':
        # This way is incompatible with Advanced ControlNet, yet.
        # guider = comfy.samplers.CFGGuider(model)
        # guider.set_conds(positive, negative)
        # guider.set_cfg(cfg)
        samples = ld.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
                                             noise_mask=noise_mask,
                                             disable_pbar=disable_pbar, seed=noise_seed)
    else:
        guider = Guider_Basic(model)
        positive = conditioning_set_values(positive, {"guidance": cfg})
        guider.set_conds(positive)
        samples = guider.sample(noise, latent_image, sampler, sigmas, denoise_mask=noise_mask, disable_pbar=disable_pbar, seed=noise_seed)

    samples = samples.to(ld.intermediate_device())

    out["samples"] = samples
    if "x0" in x0_output:
        out_denoised = latent.copy()
        out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
    else:
        out_denoised = out
    return out, out_denoised

def separated_sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                     latent_image, start_at_step, end_at_step, return_with_leftover_noise, sigma_ratio=1.0, sampler_opt=None, noise=None, callback=None, scheduler_func=None):

    if scheduler_func is not None:
        total_sigmas = scheduler_func(model, sampler_name, steps)
    else:
        if sampler_opt is None:
            total_sigmas = calculate_sigmas(model, sampler_name, scheduler, steps)
        else:
            total_sigmas = calculate_sigmas(model, "", scheduler, steps)

    sigmas = total_sigmas

    if end_at_step is not None and end_at_step < (len(total_sigmas) - 1):
        sigmas = total_sigmas[:end_at_step + 1]
        if not return_with_leftover_noise:
            sigmas[-1] = 0

    if start_at_step is not None:
        if start_at_step < (len(sigmas) - 1):
            sigmas = sigmas[start_at_step:] * sigma_ratio
        else:
            if latent_image is not None:
                return latent_image
            else:
                return {'samples': torch.zeros_like(noise)}

    if sampler_opt is None:
        impact_sampler = ksampler(sampler_name, total_sigmas)
    else:
        impact_sampler = sampler_opt

    if len(sigmas) == 0 or (len(sigmas) == 1 and sigmas[0] == 0):
        return latent_image
    
    res = sample_with_custom_noise(model, add_noise, seed, cfg, positive, negative, impact_sampler, sigmas, latent_image, noise=noise, callback=callback)

    if return_with_leftover_noise:
        return res[0]
    else:
        return res[1]

def ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None, sigma_factor=1.0, noise=None, scheduler_func=None):

    if refiner_ratio is None or refiner_model is None or refiner_clip is None or refiner_positive is None or refiner_negative is None:
        # Use separated_sample instead of KSampler for `AYS scheduler`
        # refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise * sigma_factor)[0]

        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + steps

        refined_latent = separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler,
                                          positive, negative, latent_image, start_at_step, end_at_step, False,
                                          sigma_ratio=sigma_factor, noise=noise, scheduler_func=scheduler_func)
    else:
        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + math.floor(steps * (1.0 - refiner_ratio))

        # print(f"pre: {start_at_step} .. {end_at_step} / {advanced_steps}")
        temp_latent = separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler,
                                       positive, negative, latent_image, start_at_step, end_at_step, True,
                                       sigma_ratio=sigma_factor, noise=noise, scheduler_func=scheduler_func)

        if 'noise_mask' in latent_image:
            # noise_latent = \
            #     impact_sampling.separated_sample(refiner_model, "enable", seed, advanced_steps, cfg, sampler_name,
            #                                      scheduler, refiner_positive, refiner_negative, latent_image, end_at_step,
            #                                      end_at_step, "enable")

            latent_compositor = ld.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()
            temp_latent = latent_compositor.composite(latent_image, temp_latent, 0, 0, False, latent_image['noise_mask'])[0]

        # print(f"post: {end_at_step} .. {advanced_steps + 1} / {advanced_steps}")
        refined_latent = separated_sample(refiner_model, False, seed, advanced_steps, cfg, sampler_name, scheduler,
                                          refiner_positive, refiner_negative, temp_latent, end_at_step, advanced_steps + 1, False,
                                          sigma_ratio=sigma_factor, scheduler_func=scheduler_func)

    return refined_latent

def enhance_detail(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, bbox, seed, steps, cfg,
                   sampler_name,
                   scheduler, positive, negative, denoise, noise_mask, force_inpaint,
                   wildcard_opt=None, wildcard_opt_concat_mode=None,
                   detailer_hook=None,
                   refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None,
                   refiner_negative=None, control_net_wrapper=None, cycle=1,
                   inpaint_model=False, noise_mask_feather=0, scheduler_func=None):

    if noise_mask is not None:
        noise_mask = tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
        noise_mask = noise_mask.squeeze(3)

        if noise_mask_feather > 0 and 'denoise_mask_function' not in model.model_options:
            model = DifferentialDiffusion().apply(model)[0]

    if wildcard_opt is not None and wildcard_opt != "":
        model, _, wildcard_positive = process_with_loras(wildcard_opt, model, clip)

        if wildcard_opt_concat_mode == "concat":
            positive = ld.ConditioningConcat().concat(positive, wildcard_positive)[0]
        else:
            positive = wildcard_positive
            positive = [positive[0].copy()]
            if 'pooled_output' in wildcard_positive[0][1]:
                positive[0][1]['pooled_output'] = wildcard_positive[0][1]['pooled_output']
            elif 'pooled_output' in positive[0][1]:
                del positive[0][1]['pooled_output']

    h = image.shape[1]
    w = image.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    # Skip processing if the detected bbox is already larger than the guide_size
    if not force_inpaint and bbox_h >= guide_size and bbox_w >= guide_size:
        print(f"Detailer: segment skip (enough big)")
        return None, None

    if guide_size_for_bbox:  # == "bbox"
        # Scale up based on the smaller dimension between width and height.
        upscale = guide_size / min(bbox_w, bbox_h)
    else:
        # for cropped_size
        upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    # safeguard
    if 'aitemplate_keep_loaded' in model.model_options:
        max_size = min(4096, max_size)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w = int(w * upscale)
        new_h = int(h * upscale)

    if not force_inpaint:
        if upscale <= 1.0:
            print(f"Detailer: segment skip [determined upscale factor={upscale}]")
            return None, None

        if new_w == 0 or new_h == 0:
            print(f"Detailer: segment skip [zero size={new_w, new_h}]")
            return None, None
    else:
        if upscale <= 1.0 or new_w == 0 or new_h == 0:
            print(f"Detailer: force inpaint")
            upscale = 1.0
            new_w = w
            new_h = h

    if detailer_hook is not None:
        new_w, new_h = detailer_hook.touch_scaled_size(new_w, new_h)

    print(f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}")

    # upscale
    upscaled_image = tensor_resize(image, new_w, new_h)

    cnet_pils = None
    if control_net_wrapper is not None:
        positive, negative, cnet_pils = control_net_wrapper.apply(positive, negative, upscaled_image, noise_mask)
        model, cnet_pils2 = control_net_wrapper.doit_ipadapter(model)
        cnet_pils.extend(cnet_pils2)

    # prepare mask
    if noise_mask is not None and inpaint_model:
        positive, negative, latent_image = ld.InpaintModelConditioning().encode(positive, negative, upscaled_image, vae, noise_mask)
    else:
        latent_image = to_latent_image(upscaled_image, vae)
        if noise_mask is not None:
            latent_image['noise_mask'] = noise_mask

    if detailer_hook is not None:
        latent_image = detailer_hook.post_encode(latent_image)

    refined_latent = latent_image

    # ksampler
    for i in range(0, cycle):
        if detailer_hook is not None:
            if detailer_hook is not None:
                detailer_hook.set_steps((i, cycle))

            refined_latent = detailer_hook.cycle_latent(refined_latent)

            model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, upscaled_latent2, denoise2 = \
                detailer_hook.pre_ksample(model, seed+i, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
            noise, is_touched = detailer_hook.get_custom_noise(seed+i, torch.zeros(latent_image['samples'].size()), is_touched=False)
            if not is_touched:
                noise = None
        else:
            model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, upscaled_latent2, denoise2 = \
                model, seed + i, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
            noise = None

        refined_latent = ksampler_wrapper(model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2,
                                                          refined_latent, denoise2, refiner_ratio, refiner_model, refiner_clip, refiner_positive, refiner_negative,
                                                          noise=noise, scheduler_func=scheduler_func)

    if detailer_hook is not None:
        refined_latent = detailer_hook.pre_decode(refined_latent)

    # non-latent downscale - latent downscale cause bad quality
    try:
        # try to decode image normally
        refined_image = vae.decode(refined_latent['samples'])
    except Exception as e:
        #usually an out-of-memory exception from the decode, so try a tiled approach
        refined_image = vae.decode_tiled(refined_latent["samples"], tile_x=64, tile_y=64, )

    if detailer_hook is not None:
        refined_image = detailer_hook.post_decode(refined_image)

    # downscale
    refined_image = tensor_resize(refined_image, w, h)

    # prevent mixing of device
    refined_image = refined_image.cpu()

    # don't convert to latent - latent break image
    # preserving pil is much better
    return refined_image, cnet_pils

def make_4d_mask(mask):
    if len(mask.shape) == 3:
        return mask.unsqueeze(0)

    elif len(mask.shape) == 2:
        return mask.unsqueeze(0).unsqueeze(0)

    return mask

def resize_mask(mask, size):
    mask = make_4d_mask(mask)
    resized_mask = torch.nn.functional.interpolate(mask, size=size, mode='bilinear', align_corners=False)
    return resized_mask.squeeze(0)

def tensor_paste(image1, image2, left_top, mask):
    """Mask and image2 has to be the same size"""
    _tensor_check_image(image1)
    _tensor_check_image(image2)
    _tensor_check_mask(mask)
    if image2.shape[1:3] != mask.shape[1:3]:
        mask = resize_mask(mask.squeeze(dim=3), image2.shape[1:3]).unsqueeze(dim=3)
        # raise ValueError(f"Inconsistent size: Image ({image2.shape[1:3]}) != Mask ({mask.shape[1:3]})")

    x, y = left_top
    _, h1, w1, _ = image1.shape
    _, h2, w2, _ = image2.shape

    # calculate image patch size
    w = min(w1, x + w2) - x
    h = min(h1, y + h2) - y

    # If the patch is out of bound, nothing to do!
    if w <= 0 or h <= 0:
        return

    mask = mask[:, :h, :w, :]
    image1[:, y:y+h, x:x+w, :] = (
        (1 - mask) * image1[:, y:y+h, x:x+w, :] +
        mask * image2[:, :h, :w, :]
    )
    return

def tensor_convert_rgba(image, prefer_copy=True):
    """Assumes NHWC format tensor with 1, 3 or 4 channels."""
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 4:
        return image

    if n_channel == 3:
        alpha = torch.ones((*image.shape[:-1], 1))
        return torch.cat((image, alpha), axis=-1)

    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image

    # NOTE: Similar error message as in PIL, for easier googling :P
    raise ValueError(f"illegal conversion (channels: {n_channel} -> 4)")


def tensor_convert_rgb(image, prefer_copy=True):
    """Assumes NHWC format tensor with 1, 3 or 4 channels."""
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 3:
        return image

    if n_channel == 4:
        image = image[..., :3]
        if prefer_copy:
            image = image.copy()
        return image

    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image

    # NOTE: Same error message as in PIL, for easier googling :P
    raise ValueError(f"illegal conversion (channels: {n_channel} -> 3)")

def tensor_get_size(image):
    """Mimicking `PIL.Image.size`"""
    _tensor_check_image(image)
    _, h, w, _ = image.shape
    return (w, h)

def tensor_putalpha(image, mask):
    _tensor_check_image(image)
    _tensor_check_mask(mask)
    image[..., -1] = mask[..., 0]

class DetailerForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "segs": ("SEGS", ),
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "vae": ("VAE",),
                    "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": ld.MAX_RESOLUTION, "step": 8}),
                    "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                    "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": ld.MAX_RESOLUTION, "step": 8}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (ld.SAMPLERS,),
                    "scheduler": (ld.SCHEDULERS,),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                    "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                    "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                    "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                    "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),

                    "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                   },
                "optional": {
                    "detailer_hook": ("DETAILER_HOOK",),
                    "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                   }
                }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    @staticmethod
    def do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg, sampler_name, scheduler,
                  positive, negative, denoise, feather, noise_mask, force_inpaint, wildcard_opt=None, detailer_hook=None,
                  refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None,
                  cycle=1, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None):

        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: DetailerForEach does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        image = image.clone()
        enhanced_alpha_list = []
        enhanced_list = []
        cropped_list = []
        cnet_pil_list = []

        segs = segs_scale_match(segs, image.shape)
        new_segs = []

        wildcard_concat_mode = None
        if wildcard_opt is not None:
            if wildcard_opt.startswith('[CONCAT]'):
                wildcard_concat_mode = 'concat'
                wildcard_opt = wildcard_opt[8:]
            wmode, wildcard_chooser = process_wildcard_for_segs(wildcard_opt)
        else:
            wmode, wildcard_chooser = None, None

        if wmode in ['ASC', 'DSC', 'ASC-SIZE', 'DSC-SIZE']:
            if wmode == 'ASC':
                ordered_segs = sorted(segs[1], key=lambda x: (x.bbox[0], x.bbox[1]))
            elif wmode == 'DSC':
                ordered_segs = sorted(segs[1], key=lambda x: (x.bbox[0], x.bbox[1]), reverse=True)
            elif wmode == 'ASC-SIZE':
                ordered_segs = sorted(segs[1], key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))

            else:   # wmode == 'DSC-SIZE'
                ordered_segs = sorted(segs[1], key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        else:
            ordered_segs = segs[1]

        if noise_mask_feather > 0 and 'denoise_mask_function' not in model.model_options:
            model = DifferentialDiffusion().apply(model)[0]

        for i, seg in enumerate(ordered_segs):
            cropped_image = crop_ndarray4(image.cpu().numpy(), seg.crop_region)  # Never use seg.cropped_image to handle overlapping area
            cropped_image = to_tensor(cropped_image)
            mask = to_tensor(seg.cropped_mask)
            mask = tensor_gaussian_blur_mask(mask, feather)

            is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
            if is_mask_all_zeros:
                print(f"Detailer: segment skip [empty mask]")
                continue

            if noise_mask:
                cropped_mask = seg.cropped_mask
            else:
                cropped_mask = None

            if wildcard_chooser is not None and wmode != "LAB":
                seg_seed, wildcard_item = wildcard_chooser.get(seg)
            elif wildcard_chooser is not None and wmode == "LAB":
                seg_seed, wildcard_item = None, wildcard_chooser.get(seg)
            else:
                seg_seed, wildcard_item = None, None

            seg_seed = seed + i if seg_seed is None else seg_seed

            cropped_positive = [
                [condition, {
                    k: crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                    for k, v in details.items()
                }]
                for condition, details in positive
            ]

            if not isinstance(negative, str):
                cropped_negative = [
                    [condition, {
                        k: crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                        for k, v in details.items()
                    }]
                    for condition, details in negative
                ]
            else:
                # Negative Conditioning is placeholder such as FLUX.1
                cropped_negative = negative

            if wildcard_item and wildcard_item.strip() == '[SKIP]':
                continue

            if wildcard_item and wildcard_item.strip() == '[STOP]':
                break

            orig_cropped_image = cropped_image.clone()
            enhanced_image, cnet_pils = enhance_detail(cropped_image, model, clip, vae, guide_size, guide_size_for_bbox, max_size,
                                                            seg.bbox, seg_seed, steps, cfg, sampler_name, scheduler,
                                                            cropped_positive, cropped_negative, denoise, cropped_mask, force_inpaint,
                                                            wildcard_opt=wildcard_item, wildcard_opt_concat_mode=wildcard_concat_mode,
                                                            detailer_hook=detailer_hook,
                                                            refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                                                            refiner_clip=refiner_clip, refiner_positive=refiner_positive,
                                                            refiner_negative=refiner_negative, control_net_wrapper=seg.control_net_wrapper,
                                                            cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather,
                                                            scheduler_func=scheduler_func_opt)

            if cnet_pils is not None:
                cnet_pil_list.extend(cnet_pils)

            if not (enhanced_image is None):
                # don't latent composite-> converting to latent caused poor quality
                # use image paste
                image = image.cpu()
                enhanced_image = enhanced_image.cpu()
                tensor_paste(image, enhanced_image, (seg.crop_region[0], seg.crop_region[1]), mask)  # this code affecting to `cropped_image`.
                enhanced_list.append(enhanced_image)

                if detailer_hook is not None:
                    image = detailer_hook.post_paste(image)

            if not (enhanced_image is None):
                # Convert enhanced_pil_alpha to RGBA mode
                enhanced_image_alpha = tensor_convert_rgba(enhanced_image)
                new_seg_image = enhanced_image.numpy()  # alpha should not be applied to seg_image

                # Apply the mask
                mask = tensor_resize(mask, *tensor_get_size(enhanced_image))
                tensor_putalpha(enhanced_image_alpha, mask)
                enhanced_alpha_list.append(enhanced_image_alpha)
            else:
                new_seg_image = None

            cropped_list.append(orig_cropped_image) # NOTE: Don't use `cropped_image`

            new_seg = SEG(new_seg_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
            new_segs.append(new_seg)

        image_tensor = tensor_convert_rgb(image)

        cropped_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_list.sort(key=lambda x: x.shape, reverse=True)
        enhanced_alpha_list.sort(key=lambda x: x.shape, reverse=True)

        return image_tensor, cropped_list, enhanced_list, enhanced_alpha_list, cnet_pil_list, (segs[0], new_segs)

    def doit(self, image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
             scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint, wildcard, cycle=1,
             detailer_hook=None, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None):

        enhanced_img, *_ = \
            DetailerForEach.do_detail(image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps,
                                      cfg, sampler_name, scheduler, positive, negative, denoise, feather, noise_mask,
                                      force_inpaint, wildcard, detailer_hook,
                                      cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather, scheduler_func_opt=scheduler_func_opt)

        return (enhanced_img, )

def empty_pil_tensor(w=64, h=64):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)
 
class DetailerForEachTest(DetailerForEach):
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "cropped", "cropped_refined", "cropped_refined_alpha", "cnet_images")
    OUTPUT_IS_LIST = (False, True, True, True, True)

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    def doit(self, image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
             scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint, wildcard, detailer_hook=None,
             cycle=1, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None):

        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: DetailerForEach does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        enhanced_img, cropped, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list, new_segs = \
            DetailerForEach.do_detail(image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps,
                                      cfg, sampler_name, scheduler, positive, negative, denoise, feather, noise_mask,
                                      force_inpaint, wildcard, detailer_hook,
                                      cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather, scheduler_func_opt=scheduler_func_opt)

        # set fallback image
        if len(cropped) == 0:
            cropped = [empty_pil_tensor()]

        if len(cropped_enhanced) == 0:
            cropped_enhanced = [empty_pil_tensor()]

        if len(cropped_enhanced_alpha) == 0:
            cropped_enhanced_alpha = [empty_pil_tensor()]

        if len(cnet_pil_list) == 0:
            cnet_pil_list = [empty_pil_tensor()]

        return enhanced_img, cropped, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list
