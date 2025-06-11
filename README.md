# AgentTrainer
Hope to incorporate the outputs produced by the agent’s tool invocations into the model’s reasoning as part of its reinforcement learning, achieving better performance by augmenting the reasoning process with genuine, real‐feedback signals.

## Setup 

```
cd AgentTrain
conda create --name agenttrain python=3.12
conda activate agenttrain
conda install -c conda-forge uv
uv pip install -e .
pip install flash-attn --no-build-isolation
```

## Set up vllm

```
CUDA_VISIBLE_DEVICES=0 python -m agenttrain.inference.vllm_serve --model "Qwen/Qwen2.5-VL-7B-Instruct" --tensor_parallel_size 1 --max_model_len 8192  --gpu_memory_utilization 0.9 --enable_prefix_caching True
```

Or

```
CUDA_VISIBLE_DEVICES=2 nohup python -m agenttrain.inference.vllm_serve \
  --model "Qwen/Qwen2.5-VL-7B-Instruct" \
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
CUDA_VISIBLE_DEVICES=3,4,5,6,7 nohup accelerate launch \
  --num-processes 5 \
  --config-file agenttrain/configs/zero3.yaml \
  agenttrain/main.py > training_log.log 2>&1 &
```