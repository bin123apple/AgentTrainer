CROP_TOOL_EXAMPLE = '''User: [Image_0 is displayed above] Could you identify the location of the “Close” button in this interface?

Assistant: I’d like to zoom in on the menu icon at the top-left corner to get a clearer view.  
<crop>(Image_0, (10, 20), (110, 100))</crop>

User: [Image_1 is displayed above]  # (cropped image returned)

Assistant: In this cropped image, I can see the approximate position of the “Close” button—it sits near the center of the region, slightly toward the lower-right. it’s at (45, 60).  
<answer>(Image_1, (45, 60))</answer>
'''

LOCATE_TOOL_EXAMPLE = '''User: [Image_0 is displayed above] Could you identify the location of the “Close” button in this interface?

Assistant: I’d like to use the locate tool to find the possible area where the “Close” button is located.
<locate>(Image_0,'the “Close” button')</locate>

User: [Image_1 is displayed above]  # (possible area returned)

Assistant: In this cropped image, I can see the approximate position of the “Close” button—it sits near the center of the region, slightly toward the lower-right. it’s at (45, 60).  
<answer>(Image_1, (45, 60))</answer>'''