from huggingface_hub import upload_folder, HfApi
from huggingface_hub.utils._http import HfHubHTTPError

api = HfApi()
repo_id = "Bin12345/GUI_Spotlight_venus_ckpt_120"

try:
    api.create_repo(repo_id=repo_id, private=False)
except HfHubHTTPError as e: 
    if "already exists" in str(e) or "Conflict" in str(e):
        print("仓库已存在，跳过创建步骤。")
    else:
        raise

upload_folder(
    folder_path="/home/uconn/BinLei/AgentTrainer/outputs/VG-grpo_qwen2_5vl_venus_ground-7b_2561_1ep_sft/checkpoint-120",
    repo_id=repo_id,
    commit_message="Upload trained model"
)