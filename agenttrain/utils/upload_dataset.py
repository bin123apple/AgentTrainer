from huggingface_hub import upload_folder, HfApi
from huggingface_hub.utils._http import HfHubHTTPError

api = HfApi()
repo_id = "Bin12345/5650-VG-SFT-dataset"

# Or RUN: huggingface-cli repo create 5650-VG-SFT-dataset --type dataset
try:
    api.create_repo(repo_id=repo_id, private=False)
except HfHubHTTPError as e:
    if "already exists" in str(e) or "Conflict" in str(e):
        print("仓库已存在，跳过创建步骤。")
    else:
        raise

upload_folder(
    folder_path="/mnt/data1/home/lei00126/upload_to_hf",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload dataset"
)