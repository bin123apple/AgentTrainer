from huggingface_hub import upload_folder, HfApi
from huggingface_hub.utils._http import HfHubHTTPError

api = HfApi()
repo_id = "Bin12345/qwen2_5vl_ui-tars-7b_2561_samples_1_epoch_sft_loss_0.8_beta_0_ep_0.2_0.28_grad_16_checkpoint-500"

try:
    api.create_repo(repo_id=repo_id, private=False)
except HfHubHTTPError as e:
    if "already exists" in str(e) or "Conflict" in str(e):
        print("仓库已存在，跳过创建步骤。")
    else:
        raise

upload_folder(
    folder_path="/mnt/data1/home/lei00126/AgentTrainer/outputs/VG-grpo_sft/checkpoint-500",
    repo_id=repo_id,
    commit_message="Upload trained model"
)