from huggingface_hub import upload_folder, HfApi
from huggingface_hub.utils._http import HfHubHTTPError

api = HfApi()
repo_id = "Bin12345/GUI_Spotlight_High_stage3_new_ckpt_265"

try:
    api.create_repo(repo_id=repo_id, private=False)
except HfHubHTTPError as e: 
    if "already exists" in str(e) or "Conflict" in str(e):
        print("仓库已存在，跳过创建步骤。")
    else:
        raise

upload_folder(
    folder_path="/pscratch/sd/x/xu001536/AgentTrainer/outputs/VG-grpo_qwen2_5vl_ui-tars-7b_stage3_new_125ckpt/checkpoint-140",
    repo_id=repo_id,
    commit_message="Upload trained model"
)