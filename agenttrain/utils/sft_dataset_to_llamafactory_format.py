# Updated transform_and_save function that returns the transformed data and prints the first element

import os
import json
import base64
import uuid

TOOL_PROMPT = """[Image_0 is displayed below]\nYou should use three tools to help you analyze the image and find the target coordinate:\n1. **crop**: This tool allows you to crop a specific area of the image by specifying the top-left and bottom-right coordinates of the rectangle you want to crop.\n2. **extract**: This tool allows you to extract one quarter of the image based on the specified horizontal and vertical positions (left, center, right for x-axis; top, center, bottom for y-axis).\n3. **find_color**: This tool allows you to find a specific color in the image by providing the RGB values of the target color.\nExample Usage:\n<crop>(Image_0, (10, 20), (110, 100))</crop> # Crop a rectangle from Image_0 from (10, 20) to (110, 100)\n<extract>(Image_0, left, top)</extract> # Extract the top-left quarter of Image_0\n<find_color>(Image_2, (255, 0, 0))</find_color> # Find the red color in Image_2\nBefore each tool call, please enclose your reasoning within <think>...</think> tags.\n"""

def transform_and_save(dataset, output_dir):
    """
    Transforms the input dataset and saves results to output_dir/dialogues.json,
    with decoded images saved under output_dir/images/. Returns the transformed list.
    """
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    transformed = []
    for elem in dataset:
        messages = []
        images = []
        
        for msg in elem:
            role = msg["role"]
            parts = []
            for content in msg.get("content", []):
                if content.get("type") == "text":
                    parts.append(content["text"])
                elif content.get("type") == "image_url":
                    data_url = content["image_url"]["url"]
                    _, b64data = data_url.split(",", 1)
                    img_bytes = base64.b64decode(b64data)
                    
                    filename = f"{uuid.uuid4().hex}.png"
                    filepath = os.path.join(images_dir, filename)
                    with open(filepath, "wb") as img_file:
                        img_file.write(img_bytes)
                    
                    images.append(filepath)
                    parts.append("<image>")
            
            message_text = "\n".join(parts)
            messages.append({"role": role, "content": message_text})
        
        transformed.append({"messages": messages, "images": images})
    
    # Save to dialogues.json
    out_path = os.path.join(output_dir, "dialogues.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)
    
    return transformed

if __name__ == "__main__":
    # 1. 读取原始 JSONL 数据
    import json, os

    input_path = "/mnt/data1/home/lei00126/AgentTrainer/outputs/good_completions.jsonl"
    dataset = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    # 2. 过滤：去掉任何包含 "<find_color>" 的元素
    filtered_dataset = []
    for elem in dataset:
        # 检查每条消息中的每个 text 内容
        has_find_color = False
        for msg in elem:
            for content in msg.get("content", []):
                if content.get("type") == "text" and "<find_color>" in content["text"]:
                    has_find_color = True
                    break
            if has_find_color:
                break
        if not has_find_color:
            filtered_dataset.append(elem)

    output_dir = "/mnt/data1/home/lei00126/outputs/"
    new_data = transform_and_save(filtered_dataset, output_dir)

    # 4. 打印第一个元素以供检查
    from pprint import pprint
    pprint(new_data[0])
