CROP_TOOL_DESCRIPTION = '''crop(
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int]
) -> Tuple[bytes, Tuple[int, int]]:
    """
    crop a rectangular region from an image.
    Args:
        top_left (Tuple[int, int]): The top-left corner of the cropping rectangle (x1, y1).
        bottom_right (Tuple[int, int]): The bottom-right corner of the cropping rectangle (x2, y2).
    Returns:
        The cropped image.
    """'''