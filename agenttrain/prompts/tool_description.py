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

LOCATE_TOOL_DESCRIPTION = '''locate(
    image_id: str,
    query: str
) -> Tuple[bytes, Tuple[int, int]]:
    """
    Try to locate a specific element within an image.
    Args:
        image_id (str): The ID of the image to perform `locate`, e.g., "Image_0".
        query (str): The element to locate, described in natural language.
    Returns:
        The **possible area** where the element is located, represented as a cropped image.
    """'''