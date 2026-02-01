import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
import pickle
import codecs
import io
import copy


def linspace(start, stop, step=1.0):
    """Generates evenly spaced values between start and stop with a given step."""
    return np.linspace(start, stop, int((stop - start) / step + 1))


# ────────────────────── IMAGES ──────────────────────

def resize_image_for_stable(img):
    return img.resize((math.ceil(img.size[0] / 32) * 32, math.ceil(img.size[1] / 32) * 32))

def image_array_to_data_uri(img_arr):
    img = Image.fromarray(img_arr).copy()
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_base64= base64.b64encode(buffered.getvalue()).decode("utf-8")
    return 'data:image/jpeg;base64,{}'.format(img_base64)

def uri_to_image_array(uri):
    _, image_base64 = uri.split(',', 1)
    img_data = base64.b64decode(image_base64)
    img_array = np.frombuffer(img_data, np.uint8)
    ori_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

def fixes_edges(image, mask_blur):
    mask_blur = Image.fromarray(mask_blur)
    mask_blur = np.array(resize_image_for_stable(mask_blur))

    image_blur = cv2.GaussianBlur(np.array(image), (5, 5), 0, 0)
    image_ = np.array(image)
    image_[mask_blur] = image_blur[mask_blur]
    return image_

# ────────────────────── TEXT ──────────────────────
def apply_text_layout(gen_image, binary_mask, original_image):
    """
    Applies text from original image to generated image using binary mask.
    
    Args:
        gen_image (np.ndarray): Generated image where text should be placed
        binary_mask (np.ndarray): Binary mask indicating text locations (True/1 where text exists)
        original_image (np.ndarray): Original image from which to take the text
    
    Returns:
        np.ndarray: Result image with text from original image
    """
    result_img = np.array(gen_image.copy())
    result_img[binary_mask] = original_image[binary_mask]
    return result_img

# ────────────────────── COMPOSE ──────────────────────
def compose_generated_image(gen_image, final_segments, text_layout, upscaled_image, ori_image):
    """
    Compose final image: fix edges by last mask, optionally re-apply original text, resize to original size.
    """
    final_image = gen_image
    if final_segments is not None and len(final_segments) > 0:
        final_image = fixes_edges(final_image, final_segments[-1])
    if text_layout is not None and len(text_layout) != 0:
        final_image = apply_text_layout(final_image, text_layout, upscaled_image)
    height, width, channels = ori_image.shape
    res_image_resized = cv2.resize(final_image, (width, height))
    return res_image_resized

# ────────────────────── POINTS ──────────────────────

def scale_points(points, original_image, upscaled_image):
    orig_height, orig_width = original_image.shape[:2]
    upscaled_height, upscaled_width = upscaled_image.shape[:2]
    
    scale_x = upscaled_width / orig_width
    scale_y = upscaled_height / orig_height
    
    scaled_points = []
    for point in points:
        new_x = int(point[0] * scale_x)
        new_y = int(point[1] * scale_y)
        scaled_points.append((new_x, new_y))
    
    return scaled_points

# ────────────────────── MASKS ──────────────────────

def mask_to_rle(masks):
    b, h, w = masks.shape
    masks = masks.transpose((0, 2, 1)).reshape(masks.shape[0], -1)
    diff = masks[:, 1:] ^ masks[:, :-1]
    change_indices = np.stack(diff.nonzero(), axis=-1)
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = np.concatenate(
            [
                np.array([0], dtype=cur_idxs.dtype),
                cur_idxs + 1,
                np.array([h * w], dtype=cur_idxs.dtype),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if masks[i, 0] == 0 else [0]
        counts.extend(btw_idxs.tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def expand_mask(mask, iteration=1):
    kernel = np.ones((3, 3), np.uint8)
    res = cv2.dilate(np.uint8(mask), kernel, iterations=iteration).astype(bool)
    return res

def calc_iou(seg1, seg2):
    intersection = np.logical_and(seg1, seg2).sum()
    union = np.logical_or(seg1, seg2).sum()
    return intersection / union


def segment_intesection(seg1, seg2):
    iou = calc_iou(seg1, seg2)
    return True if iou > 0.01 else False

def area_from_rle(rle) -> int:
    return sum(rle["counts"][1::2])

# ────────────────────── BOXES ──────────────────────

def bbox_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    :param box1: First bounding box in format [y1, x1, y2, x2]
    :param box2: Second bounding box in format [y1, x1, y2, x2]
    :return: IoU value between 0 and 1
    """
    y1 = max(box1[0], box2[0])
    x1 = max(box1[1], box2[1])
    y2 = min(box1[2], box2[2])
    x2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

# ────────────────────── 4 TESTS ──────────────────────
def show_with_mask(ori_img, masks):
    img_rgba = np.concatenate([ori_img, np.ones((*ori_img.shape[:2], 1), dtype=np.uint8) * 255], axis=-1)

    for m in masks:
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        color = (color_mask[:3] * 255).astype(np.uint8)
        alpha = int(color_mask[3] * 255)

        mask = np.zeros_like(ori_img, dtype=np.uint8)
        mask[m] = color
        alpha_mask = np.zeros((*ori_img.shape[:2], 1), dtype=np.uint8)
        alpha_mask[m] = alpha

        mask_rgba = np.concatenate([mask, alpha_mask], axis=-1)

        img_rgba = img_rgba * (1 - alpha_mask / 255) + mask_rgba * (alpha_mask / 255)

    return img_rgba.astype(np.uint8)

def save_image(image_array, file_path):
    rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    with open(file_path, "wb") as f:
        success, encoded_image = cv2.imencode(".png", rgb_image)
        if success:
            f.write(encoded_image.tobytes())

def draw_image_with_points(image, points):
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)
    for point in points:
        draw.ellipse([point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5], fill="red", outline="red")

    return np.array(image_pil)

# ────────────────────── TEXT REMOVAL ──────────────────────

def prepare_bboxes_to_merge(bboxes, shift):
    new_bboxes = []
    for bb in bboxes:
        x, y = bb[0] - shift, bb[1] - shift
        x1, y1 = bb[2] + shift, bb[3] + shift
        new_bboxes.append([y, x, y1, x1])
    return np.array(new_bboxes)

def merge_bboxes(bboxes, delta_x=0.1, delta_y=0.1):
    bboxes = sorted(bboxes, key=lambda x: x[1])
    tmp_bbox = None
    while True:
        nb_merge = 0
        used = []
        new_bboxes = []
        for i, b in enumerate(bboxes):
            for j, b_ in enumerate(bboxes):
                if i in used or j <= i:
                    continue
                bmargin = [
                    b[0] - (b[2] - b[0]) * delta_x,
                    b[1] - (b[3] - b[1]) * delta_y,
                    b[2] + (b[2] - b[0]) * delta_x,
                    b[3] + (b[3] - b[1]) * delta_y,
                ]
                b_margin = [
                    b_[0] - (b_[2] - b_[0]) * delta_x,
                    b_[1] - (b_[3] - b_[1]) * delta_y,
                    b_[2] + (b_[2] - b_[0]) * delta_x,
                    b_[3] + (b_[3] - b_[1]) * delta_y,
                ]
                if bbox_iou(bmargin, b_margin) or bbox_iou(b_margin, bmargin):
                    tmp_bbox = [min(b[0], b_[0]), min(b[1], b_[1]), max(b_[2], b[2]), max(b[3], b_[3])]
                    used.append(j)
                    nb_merge += 1
                if tmp_bbox:
                    b = tmp_bbox
            if tmp_bbox:
                new_bboxes.append(tmp_bbox)
            elif i not in used:
                new_bboxes.append(b)
            used.append(i)
            tmp_bbox = None
        if nb_merge == 0:
            break
        bboxes = copy.deepcopy(new_bboxes)

    return np.array(new_bboxes)

def inpaint_by_mask(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, x + w, y + h])
    prepared_bboxes = prepare_bboxes_to_merge(bboxes, shift=3)
    merged_bboxes = merge_bboxes(prepared_bboxes, delta_x=0.1, delta_y=0.1)
    inpaint_mask = np.zeros_like(mask)
    for bbox in merged_bboxes:
        y1, x1, y2, x2 = map(int, bbox)
        cv2.rectangle(inpaint_mask, (x1, y1), (x2, y2), 255, -1)
    image_inpainted = cv2.inpaint(image.astype(np.uint8), inpaint_mask, 15, cv2.INPAINT_TELEA)
    return image_inpainted