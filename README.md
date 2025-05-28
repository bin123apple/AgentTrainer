# AgentTrainer
We incorporate the outputs produced by the agent’s tool invocations into the model’s reasoning as part of its reinforcement learning, achieving better performance by augmenting the reasoning process with genuine, real‐feedback signals.

## Setup 

1. Setup enviroment
```
cd AgentTrain
conda create --name agenttrain python=3.12
conda activate agenttrain
conda install -c conda-forge uv
uv pip install -e .
```