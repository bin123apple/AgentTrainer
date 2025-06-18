from huggingface_hub import upload_folder, HfApi
from huggingface_hub.utils._http import HfHubHTTPError

api = HfApi()
repo_id = "Bin12345/Qwen-2.5B-VL-7B-VG-sft-2633-steps"

try:
    api.create_repo(repo_id=repo_id, private=False)
except HfHubHTTPError as e:
    if "already exists" in str(e) or "Conflict" in str(e):
        print("仓库已存在，跳过创建步骤。")
    else:
        raise

upload_folder(
    folder_path="/home/uconn/BinLei/LLaMA-Factory/saves/qwen2_5vl-7b/full/sft",
    repo_id=repo_id,
    commit_message="Upload trained model"
)