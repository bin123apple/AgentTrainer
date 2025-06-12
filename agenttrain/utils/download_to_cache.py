import os
from huggingface_hub import login, snapshot_download
from datasets import load_dataset

def main():
    # 1. 读取 Hugging Face 访问令牌
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise RuntimeError("请先在环境变量 HUGGINGFACE_TOKEN 中设置你的 Hugging Face 访问令牌")

    # 2. 登录 Hugging Face
    login(token=hf_token)

    # 3. 并行下载整个数据集到指定目录
    cache_dir = "/mnt/data1/huggingface/datasets"
    local_dir = snapshot_download(
        repo_id="osunlp/UGround-V1-Data-Box",
        repo_type="dataset",
        cache_dir=cache_dir,
        use_auth_token=hf_token,
        max_workers=16           # 并行 download 线程数，可根据网络/CPU 调整
    )
    print(f"数据集已下载至：{local_dir}")

    # # 4. 从本地目录加载数据集
    # dataset = load_dataset(local_dir, cache_dir=cache_dir)
    # print(dataset)

if __name__ == "__main__":
    main()

