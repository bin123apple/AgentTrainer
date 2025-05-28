SIMPLE_PROMPT = """
Respond in the following format, using careful step-by-step reasoning.

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

CODE_PROMPT = """\
Given a math problem, use step-by-step reasoning and code execution to solve the problem. 

For each step:
1. Think through your reasoning inside <reasoning> tags
2. Write Python scripts inside <code> tags to work out calculations
   - Functions and variables do not persist across <code> calls and should be redefined each time
   - Scripts should be written in Python 3.10+ syntax, and should run in under 10 seconds
   - Any desired outputs should be printed using print() statements
   - You may import numpy, scipy, and sympy libraries for your calculations
3. You will see the output from print() statements in your code in <output> tags
4. Continue until you can give the final answer inside <answer> tags
"""

DEFAULT_TOOL_PROMPT_TEMPLATE = """You have access to the following tools to help solve problems:

{tool_descriptions}

For each step:
1. Think through your reasoning inside <reasoning> tags
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
3. You will see the tool's output inside <result> tags
4. Continue until you can give the final answer inside <answer> tags

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed.
"""

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
You may use the crop tool to help you analyze and hone in on the target coordinate by placing the tool call inside <crop>...</crop> tags; 
each time you call the crop tool, I will return the resulting cropped image to you. 
In the end, you must place your selected coordinate inside <answer>...</answer> tags.

The crop function is used like this: 

{tool_descriptions}

and here is an example of its use: 

{tool_example}

Please note:
1. You may call the crop tool multiple times if needed.
2. Each crop is always taken relative to the initial image, not to any previously cropped image.
3. Your final coordinate must also be given relative to the initial image.
4. The <answer>...</answer> tags should contain only your final coordinate."""