from PIL import Image
import io
from typing import Tuple, Any

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
          - On success: (PNG bytes of cropped image, descriptive message)
          - On failure: (None, detailed error message)
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    width, height = img.size

    # 边界检查并收集所有错误
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
            f"Invalid crop coordinates: {detail}. "
            f"Image size: width={width}, height={height}; "
            f"Requested: top_left=({x1}, {y1}), bottom_right=({x2}, {y2})."
        )
        return None, msg

    # 执行裁剪
    cropped = img.crop((x1, y1, x2, y2))
    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG")
    data = buffer.getvalue()

    # 生成带微秒级时间戳的文件名并保存
    # ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # filename = f"output_{ts}.png"
    # cropped.save(filename, format="PNG")

    # 构造描述信息
    message = (
        f"Cropped a region of size {x2-x1}×{y2-y1} pixels "
        f"from the original image ({width}×{height}), "
        f"located at top-left ({x1}, {y1}) and bottom-right ({x2}, {y2})."
    )
    return data, message