import io
import os
from datetime import datetime
from typing import Tuple, Any
from PIL import Image
from typing import List

def crop(
    img: Image.Image,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int]
) -> Tuple[Any, str]:
    """
    crop a rectangular region from an image.
    Args:
        img (Image.Image): The image to crop.
        top_left (Tuple[int, int]): The top-left corner of the cropping rectangle (x1, y1).
        bottom_right (Tuple[int, int]): The bottom-right corner of the cropping rectangle (x2, y2).
    Returns:
        Tuple[bytes, str]:
            cropped image as PNG bytes,
            message -> As text describing the crop operation
            offset -> For calculating the coordinates relative to the original image
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    width, height = img.size

    # Boundary checks
    errors = []
    if x1 < 0:
        errors.append(f"x1 ({x1}) < 0")
    if y1 < 0:
        errors.append(f"y1 ({y1}) < 0")
    if x2 > width:
        errors.append(f"x2 ({x2}) > image width ({width})")
    if y2 > height:
        errors.append(f"y2 ({y2}) > image height ({height})")
    if x2 <= x1:
        errors.append(f"x2 ({x2}) ≤ x1 ({x1})")
    if y2 <= y1:
        errors.append(f"y2 ({y2}) ≤ y1 ({y1})")

    if errors:
        detail = "; ".join(errors)
        msg = (
            "<|Tool_Error|>: "
            f"Invalid crop coordinates: {detail}. "
            f"Image size: width={width}, height={height}; "
            f"Requested: top_left=({x1}, {y1}), bottom_right=({x2}, {y2})."
        )
        return None, msg, None

    # Ensure minimum dimensions
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w < 28 or crop_h < 28:
        return None, (
            "<|Tool_Error|>: "
            f"Crop size too small: width={crop_w}, height={crop_h}. "
            "Both crop_w and crop_h must be at least 28 pixels."
        ), None

    # Adjust for edge case of width or height exactly 3
    if crop_w == 3:
        if x2 < width:
            x2 += 1
        elif x1 > 0:
            x1 -= 1
    if crop_h == 3:
        if y2 < height:
            y2 += 1
        elif y1 > 0:
            y1 -= 1

    # Perform crop
    cropped = img.crop((x1, y1, x2, y2))
    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG")
    data = buffer.getvalue()

    # # Save backup
    # os.makedirs("backup", exist_ok=True)
    # filename = f"backup/output_(({x1},{y1}),({x2},{y2})).png"
    # cropped.save(filename, format="PNG")

    # Descriptive message
    new_w, new_h = cropped.size
    message = f"Cropped a region of size {new_w}×{new_h} pixels."
    return data, message, top_left


# # ---------- 参数可按需求调节 ----------
# MODEL_NAME = "ViT-B-32"          # 轻量版；ViT-L/14 精度更好
# PRETRAIN   = "openai"
# WIN_SIZE   = 224                 # 滑窗尺寸
# STRIDE     = 112                 # 步幅 = 50% 重叠
# TOP_K      = 5                   # 返回候选框数量
# SIM_THRES  = 0.25                # 相似度阈值 (0~1)

# # ---------- 预加载模型 ----------
# _device = "cuda" if torch.cuda.is_available() else "cpu"
# model, _, preprocess = open_clip.create_model_and_transforms(
#         MODEL_NAME, pretrained=PRETRAIN, device=_device)
# model.eval()
# tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# @torch.no_grad()
# def scan(
#     image: Image.Image,
#     text_query: str,
#     win_size: int = WIN_SIZE,
#     stride: int = STRIDE,
#     sim_thres: float = SIM_THRES,
# ):
#     """
#     输入 : PIL.Image + 文本描述
#     输出 : Tuple[bytes, str]:
#             cropped image as PNG bytes,
#             message -> As text describing the crop operation
#             offset -> For calculating the coordinates relative to the original image
#     """
#     W, H = image.size

#     # 1) 文本向量
#     text_feat = model.encode_text(tokenizer([text_query]).to(_device)).float()
#     text_feat /= text_feat.norm(dim=-1, keepdim=True)

#     bgr  = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#     best_box, best_score = None, -1.0

#     # 2) 枚举所有起点（含补尾块）
#     try:
#         y_starts = list(range(0, H - win_size + 1, stride))
#         if y_starts[-1] + win_size < H:
#             y_starts.append(H - win_size)

#         x_starts = list(range(0, W - win_size + 1, stride))
#         if x_starts[-1] + win_size < W:
#             x_starts.append(W - win_size)
#     except IndexError:
#         msg = (f"<|Tool_Error|>, The image is too small ({W}×{H}) to apply `scan` function. "
#                f"Please ensure the image is at least {win_size}×{win_size} pixels if you would like to use the `scan` function.")
#         return None, msg, None

#     # 3) 滑窗 & 取分最高者
#     for y in y_starts:
#         for x in x_starts:
#             tile = bgr[y:y + win_size, x:x + win_size]
#             img_tensor = preprocess(Image.fromarray(cv2.cvtColor(
#                 tile, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(_device)

#             img_feat = model.encode_image(img_tensor).float()
#             img_feat /= img_feat.norm(dim=-1, keepdim=True)
#             sim = (img_feat @ text_feat.T).item()

#             if sim >= sim_thres and sim > best_score:
#                 best_score = sim
#                 best_box   = (x, y, x + win_size, y + win_size)

#     # 4) 结果处理
#     if best_box is None:
#         # 未找到
#         msg = (f"No window exceeds similarity threshold {sim_thres}.")
#         return None, msg, None

#     # 做 crop
#     x1, y1, x2, y2 = best_box
#     cropped = image.crop(best_box)
#     buffer  = io.BytesIO()
#     cropped.save(buffer, format="PNG")
#     data = buffer.getvalue()

#     new_w, new_h = cropped.size
#     msg = (f"Cropped a region of size {new_w}×{new_h} pixels.\n"
#            "NOTE: The `scan` function returns the **possible** area where the element is located, "
#            "so you may need to verify its correctness yourself.")

#     # （如需本地备份，取消下面两行注释）
#     # os.makedirs("backup", exist_ok=True)
#     # cropped.save(f"backup/output_(({x1},{y1}),({x2},{y2})).png", format="PNG")

#     return data, msg, (x1, y1)

def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0: return 0.0
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (area1 + area2 - inter)


from typing import Tuple, Optional, Union
from PIL import Image
import os, io

_X_OPTIONS = {"left", "center", "right"}
_Y_OPTIONS = {"top", "center", "bottom"}
_MIN_SIDE   = 28         # cropped width / height must be ≥ 28 px

def extract(
    img_input: Union[str, Image.Image],
    x_pos: str,
    y_pos: str
) -> Tuple[Optional[bytes], str, Optional[Tuple[int, int]]]:
    """
    Extract one-quarter of an image (½ width × ½ height).

    Returns:
        data (bytes | None)        : PNG bytes of the cropped image, or None on error
        message (str)              : success / error message
        offset  (Tuple[int,int] | None): (x0, y0) of the crop in the original image
    """
    # 1. load image ------------------------------------------------------
    if isinstance(img_input, Image.Image):
        img = img_input
    elif isinstance(img_input, str):
        if not os.path.isfile(img_input):
            return None, f"File not found: {img_input}", None
        try:
            img = Image.open(img_input)
        except Exception as e:
            return None, f"Could not open image: {e}", None
    else:
        return None, "Invalid img_input: must be file path or PIL.Image.Image", None

    # 2. validate positions ---------------------------------------------
    x_pos, y_pos = x_pos.lower(), y_pos.lower()
    if x_pos not in _X_OPTIONS or y_pos not in _Y_OPTIONS:
        return None, (
            f"x_pos must be one of {_X_OPTIONS}, "
            f"y_pos must be one of {_Y_OPTIONS}."
        ), None

    W, H = img.size
    half_w, half_h = W // 2, H // 2              # target crop size

    # --- NEW: minimum-size check ---------------------------------------
    if half_w < _MIN_SIDE or half_h < _MIN_SIDE:
        return None, (
            "<|Tool_Error|>: "
            f"Crop size too small: width={half_w}, height={half_h}. "
            f"Both crop_w and crop_h must be at least {_MIN_SIDE} pixels."
        ), None

    # 3. compute offset --------------------------------------------------
    x0 = 0 if x_pos == "left"   else (W - half_w) // 2 if x_pos == "center" else W - half_w
    y0 = 0 if y_pos == "top"    else (H - half_h) // 2 if y_pos == "center" else H - half_h

    # 4. crop & serialize -----------------------------------------------
    crop_box = (x0, y0, x0 + half_w, y0 + half_h)
    cropped  = img.crop(crop_box)

    with io.BytesIO() as buf:
        cropped.save(buf, format="PNG")
        data = buf.getvalue()

    message = f"Cropped a region of size {half_w}×{half_h}."
    return data, message, (x0, y0)


import os, io, cv2, numpy as np
from typing import Tuple, Optional, Union
from PIL import Image

# ΔE（CIE76）------------------------------------------------------------
def _delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    return float(np.linalg.norm(lab1.astype(np.float32) - lab2.astype(np.float32)))

# 主函数 ---------------------------------------------------------------
def find_color(
    img_input : Union[str, Image.Image],
    target_rgb: Tuple[int, int, int],
) -> Tuple[Optional[bytes], str, Optional[Tuple[int, int]]]:
    """
    1. 用 10×10 小块在整张图上滑动（步长 10），找 ΔE 最小的小块；
    2. 以该小块中心为圆心取 200×200 窗口（必要时向内平移使其完全落在图内）；
    3. 返回窗口的 PNG 字节流、提示信息、窗口左上角 (x0, y0)。
    """
    # ---------- 读取图像 ---------- #
    if isinstance(img_input, Image.Image):
        # PIL → ndarray (BGR)
        img_bgr = cv2.cvtColor(np.asarray(img_input), cv2.COLOR_RGB2BGR)
    elif isinstance(img_input, str):
        if not os.path.isfile(img_input):
            return None, f"File not found: {img_input}", None
        img_bgr = cv2.imread(img_input)
        if img_bgr is None:
            return None, f"Could not open image: {img_input}", None
    else:
        return None, "Invalid img_input: must be file path or PIL.Image.Image", None

    h, w = img_bgr.shape[:2]

    # ---------- 自适应窗口大小 ---------- #
    ws = 200                          # 目标窗口
    if min(h, w) < ws:                # 若图像最短边 < 200
        ws = min(h, w)                #   用最短边替代
    half = ws // 2                    # 半径 (= ws / 2)

    # ---------- 颜色准备 ---------- #
    tgt_lab = cv2.cvtColor(
        np.uint8([[target_rgb[::-1]]]),   # BGR
        cv2.COLOR_BGR2LAB
    )[0, 0]

    # ---------- 滑动 10×10 小块 ---------- #
    best = {"delta_e": 1e9}
    sp, stride = 10, 10
    for y in range(0, h - sp + 1, stride):
        for x in range(0, w - sp + 1, stride):
            patch = img_bgr[y:y+sp, x:x+sp]
            mean_lab = cv2.cvtColor(
                patch.mean(axis=(0,1), dtype=np.float32).reshape(1,1,3).astype(np.uint8),
                cv2.COLOR_BGR2LAB
            )[0, 0]
            de = _delta_e_cie76(mean_lab, tgt_lab)
            if de < best["delta_e"]:
                best.update({"delta_e": de, "center": (x + sp//2, y + sp//2)})

    # ---------- 定位 200×200 窗口 ---------- #
    cx, cy = best["center"]
    ws, half = 200, 100
    x0 = max(0, min(cx - half, w - ws))
    y0 = max(0, min(cy - half, h - ws))
    window = img_bgr[y0:y0+ws, x0:x0+ws]

    # ---------- PNG 序列化 ---------- #
    success, buf = cv2.imencode(".png", window)
    if not success:
        return None, "Failed to encode PNG.", None
    data = buf.tobytes()

    msg = f"Cropped a region of size {ws}×{ws} matching target RGB {target_rgb}."
    return data, msg, (x0, y0)