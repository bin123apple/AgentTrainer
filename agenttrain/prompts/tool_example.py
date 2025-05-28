CROP_TOOL_EXAMPLE = '''User: [ImageDisplayed] Could you identify the location of the “Close” button in this interface?

Assistant: I’d like to zoom in on the menu icon at the top-left corner to get a clearer view.  
<crop>crop((10, 20), (110, 100))</crop>

User: <<ImageDisplayed>>  # (cropped image returned)

Assistant: In this cropped image, I can see the approximate position of the “Close” button—it sits near the center of the region, slightly toward the lower-right. Converting back to the original image, it’s approximately at (45, 60).  
<answer>(45, 60)</answer>'''