# AgentTrainer
Hope to incorporate the outputs produced by the agent’s tool invocations into the model’s reasoning as part of its reinforcement learning, achieving better performance by augmenting the reasoning process with genuine, real‐feedback signals.

## SFT

## Collect data
```
CUDA_VISIBLE_DEVICES=4,5 nohup python agenttrain/sft/sft_data_generation.py > agenttrain_sft_data.log 2>&1 &
```

## SFT Train
Install llama_factory based on their readme. Change 
1. `dataset` (dataset_info file) 
2. `dataloader_num_workers` in yaml to 0
3. `cutoff_len`
4. `max_samples`
Then Run:
```
tmux new -s llama_train
source /mnt/data1/home/lei00126/miniconda3/etc/profile.d/conda.sh
conda activate llama_factory
llamafactory-cli train examples/train_full/qwen2_5vl_full_sft.yaml
ctrl+b d 
```

Check background task:
```
tmux ls
tmux attach -t agenttrain
```




## RL

## Setup 

```
cd AgentTrain
conda create --name agenttrain python=3.12
conda activate agenttrain
conda install -c conda-forge uv
uv pip install -e .
pip install flash-attn --no-build-isolation
```

## prepare dataset
```
python agenttrain/utils/download_to_cache.py
python agenttrain/utils/data_collection_save.py
```


## Set up vllm

```
CUDA_VISIBLE_DEVICES=0 python -m agenttrain.inference.vllm_serve --model "Qwen/Qwen2.5-VL-7B-Instruct" --tensor_parallel_size 1 --max_model_len 8192  --gpu_memory_utilization 0.9 --enable_prefix_caching True
```

Or

```
CUDA_VISIBLE_DEVICES=4 nohup python -m agenttrain.inference.vllm_serve \
  --model "/mnt/data1/home/lei00126/LLaMA-Factory/saves/qwen2_5vl_ui-tars-7b/full/sft" \
  --tensor_parallel_size 1 \
  --max_model_len 8192 \
  --gpu_memory_utilization 0.95 \
  --enable_prefix_caching True \
  --host 0.0.0.0 \
  --port 8888 > vllm_log.log 2>&1 &
```

## Train

```
CUDA_VISIBLE_DEVICES=5,6,7 accelerate launch --num-processes 3 --config-file agenttrain/configs/zero3.yaml agenttrain/main.py
```
OR

```
export CUDA_VISIBLE_DEVICES=5,6,7
LOGDIR=logs/$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"

nohup accelerate launch \
  --num_processes 3 \
  --config_file agenttrain/configs/zero3.yaml \
  --tee 3 \
  agenttrain/main.py \
  > "$LOGDIR/master.log" 2>&1 &
echo "Started! Logs in $LOGDIR"
```