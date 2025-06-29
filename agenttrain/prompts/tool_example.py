CROP_TOOL_EXAMPLE = '''User: [Image_0 is displayed above] Could you identify the location of the “Close” button in this interface?

Assistant: <think>I’d like to zoom in on the menu icon at the top-left corner to get a clearer view.</think>  
<crop>(Image_0, (10, 20), (110, 100))</crop>

User: [Image_1 is displayed above]  # (cropped image returned)

Assistant: <think>In this cropped image, I can see the approximate position of the “Close” button—it sits near the center of the region, slightly toward the lower-right. it’s at (45, 60).</think>  
<answer>(Image_1, (45, 60))</answer>
'''

SCAN_TOOL_EXAMPLE = '''User: [Image_0 is displayed above] Could you identify the location of the “Close” button in this interface?

Assistant: <think>I’d like to use the scan tool to find the possible area where the “Close” button is located.</think>
<scan>(Image_0,'the “Close” button')</scan>

User: [Image_1 is displayed above]  # (possible area returned)

Assistant: <think>In this cropped image, I can see the approximate position of the “Close” button—it sits near the center of the region, slightly toward the lower-right. it’s at (45, 60).</think>  
<answer>(Image_1, (45, 60))</answer>'''

EXTRACT_TOOL_EXAMPLE = '''User: [Image_0 is displayed above] Could you locate the “Submit” button in this form?

Assistant: <think>I’ll zoom into the upper-left quarter first; the button text should be somewhere around there.</think>  
<extract>(Image_0, left, top)</extract>

User: [Image_1 is displayed above]  # (extracted quarter returned)

Assistant: <think>In this quadrant I can now see the “Submit” button—it’s roughly at the lower-right of this region, around (30, 45).</think>  
<answer>(Image_1, (30, 45))</answer>
'''

FIND_COLOR_TOOL_EXAMPLE = '''User: [Image_0 is displayed above] Could you identify the location of the “Close” button in this interface?

Assistant: <think>Now I suspect the “Close” button is the red circle icon in this region. I will use `find_color` to locate the red element.</think>
<find_color>(Image_2, (255, 0, 0))</find_color>

User: [Image_1 is displayed above]  # (best color-matched region)

Assistant: <think>The “Close” button is clearly visible in this region and located approximately at (45, 60).</think>  
<answer>(Image_1, (45, 60))</answer>'''

MERGE_TOOL_EXAMPLE = '''User: [Image_0 is displayed above] Could you identify the location of the “Close” button in this interface?

Assistant: <think>I’d like to zoom in on the menu icon at the top-left corner to get a clearer view.</think>  
<crop>(Image_0, (10, 20), (110, 100))</crop>

User: [Image_1 is displayed above]  # (cropped image returned)xw

Assistant: <think>In this cropped image, I can not see the “Close” button, I will use the `extract` tool to find the possible area where the “Close” button is located.</think>
<extract>(Image_0, left, top)</extract>

User: [Image_2 is displayed above]  # (possible area returned)

Assistant: <think>In this cropped image, I can see the approximate position of the “Close” button—it sits near the center of the region, slightly toward the lower-right. it’s at (45, 60).</think>  
<answer>(Image_2, (45, 60))</answer>'''