CROP_TOOL_PROMPT_TEMPLATE = """You have access to the following tools to help solve problems:

{tool_descriptions}
For each step:
For each visual localization step, if you feel it’s necessary to crop out a region to inspect details, please call:
crop_region(
    img: Image.Image,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int]
)
and wrap that call in <crop>...</crop> tags—you will then see the cropped area. Finally, provide the coordinates relative to the original full image and wrap them in <answer>...</answer> tags.
Please follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""

CROP_SYSTEM_PROMPT = """Your goal is to accurately provide a coordinate point based on the user’s description and the initial image they supplied. 
You may use the crop and locate tool to help you analyze and hone in on the target coordinate by placing the tool call inside <crop>...</crop> and <locate>...</locate> tags; 
each time you call the tool, I will return the result to you. 
In the end, you must place your selected coordinate inside <answer>...</answer> tags.

The tool functions are used like this: 

{tool_descriptions}

and here are examples of their use: 

{tool_example}

Please note:
1. You may call the tool multiple times if needed.
2. Do NOT include parameter names (e.g., image_id, top_left, bottom_right) in your tool calls.
For example, use <crop>(Image_0, (10, 20), (110, 100))</crop> instead of <crop image_id="Image_0" top_left="(10, 20)" bottom_right="(110, 100)"></crop>.
3. The `locate` tool can only help you to find the **possible** area where the element is located. There's a chance that the element won't appear in the image returned by the `locate` command, so you'll need to verify it yourself.
4. In the end you must provide the coordinate in the format <answer>(Image_id, (x, y))</answer>, where the (x, y) values are relative to the image corresponding to Image_id, not to the original image."""