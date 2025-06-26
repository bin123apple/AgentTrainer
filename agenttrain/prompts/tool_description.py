CROP_TOOL_DESCRIPTION = '''crop(
    image_id: str,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int]
) -> Tuple[bytes, Tuple[int, int]]:
    """
    crop a rectangular region from an image.
    Args:
        image_id (str): The ID of the image to `crop`, e.g., "Image_0".
        top_left (Tuple[int, int]): The top-left corner of the cropping rectangle (x1, y1).
        bottom_right (Tuple[int, int]): The bottom-right corner of the cropping rectangle (x2, y2).
    Returns:
        The cropped image.
    """
'''

SCAN_TOOL_DESCRIPTION = '''scan(
    image_id: str,
    query: str
) -> Tuple[bytes, Tuple[int, int]]:
    """
    Try to scan a specific element within an image.
    Args:
        image_id (str): The ID of the image to perform `scan`, e.g., "Image_0".
        query (str): The element to scan, described in natural language.
    Returns:
        The **possible area** where the element is located, represented as a cropped image.
    """'''

EXTRACT_TOOL_DESCRIPTION = '''extract(
    image_id: str,
    x_pos: Literal["left", "center", "right"],
    y_pos: Literal["top", "center", "bottom"]
) -> Tuple[Optional[bytes], str, Optional[Tuple[int, int]]]:
    """
    Extract one quarter of an image (half the width and half the height) based on the
    specified horizontal and vertical positions.

    Args:
        image_id (str): The ID of the image to extract from (e.g., "Image_0").
        x_pos (Literal["left", "center", "right"]):
            Which half-column to take along the x-axis:
            - "left"   : left half
            - "center" : center half
            - "right"  : right half
        y_pos (Literal["top", "center", "bottom"]):
            Which half-row to take along the y-axis:
            - "top"    : top half
            - "center" : center half
            - "bottom" : bottom half

    Returns:
        Tuple containing:
            1. bytes | None:
               The extracted image in bytes if successful; otherwise, None.
            2. str:
               A message describing the outcome (success or error).
            3. Tuple[int, int] | None:
               (x_offset, y_offset) of the top-left corner of the extracted region
               relative to the original image, or None if extraction failed.
    """
'''
