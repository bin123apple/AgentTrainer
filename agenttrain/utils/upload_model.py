from huggingface_hub import upload_folder, HfApi
from huggingface_hub.utils._http import HfHubHTTPError

api = HfApi()
repo_id = "Bin12345/qwen2_5vl_ui-tars-7b_2561sp_1_ep_sft_1_beta_0_ep_NaNa_grad16_cpkt-400_sft_0.1_100_sft_0.1_160_h"

try:
    api.create_repo(repo_id=repo_id, private=False)
except HfHubHTTPError as e: 
    if "already exists" in str(e) or "Conflict" in str(e):
        print("仓库已存在，跳过创建步骤。")
    else:
        raise

upload_folder(
    folder_path="/mnt/data1/home/lei00126/AgentTrainer/outputs/VG-grpo_qwen2_5vl_ui-tars-7b_2561_samples_1_epoch_rl/checkpoint-60",
    repo_id=repo_id,
    commit_message="Upload trained model"
)