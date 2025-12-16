# Streaming Deep Reinforcement Learning

# RL class project 
This project is based on the [streaming RL paper](https://arxiv.org/abs/2410.14606)
to use wandb use add flag ```--track=1```, you need to have a wandb account check the guide in the https://docs.wandb.ai/models/quickstart

# rerunning experiments 
To rerun the experiments for each empirical question, use the following commands:
For question 1 (Q1: effect of function approximation on convergence in the online streaming setting?):
```
bash run_stream_q_learning 
```
For question 4 (Q4: How does streaming perform in partially observed settings?)
```
bash run_stream_q_partial_obs
```