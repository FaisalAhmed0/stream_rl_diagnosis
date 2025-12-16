# Streaming Deep Reinforcement Learning

# RL class project 

## Rerunning experiments

Commands to reproduce the full set of runs (seeds 0, 1, 2).

python stream_q_continuous.py --debug --env HalfCheetah-v4 --track=1 --total_steps=10000000 --seed=0 --gradient_steps_per_step=1
python stream_q_continuous.py --debug --env HalfCheetah-v4 --track=1 --total_steps=10000000 --seed=1 --gradient_steps_per_step=1
python stream_q_continuous.py --debug --env HalfCheetah-v4 --track=1 --total_steps=10000000 --seed=2 --gradient_steps_per_step=1

python stream_ac_continuous.py --debug --env HalfCheetah-v4 --track=1 --total_steps=10000000 --seed=0 --gradient_steps_per_step=1
python stream_ac_continuous.py --debug --env HalfCheetah-v4 --track=1 --total_steps=10000000 --seed=1 --gradient_steps_per_step=1
python stream_ac_continuous.py --debug --env HalfCheetah-v4 --track=1 --total_steps=10000000 --seed=2 --gradient_steps_per_step=1

python stream_q_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=10000000 --seed=0 --gradient_steps_per_step=1
python stream_q_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=10000000 --seed=1 --gradient_steps_per_step=1
python stream_q_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=10000000 --seed=2 --gradient_steps_per_step=1

python stream_ac_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=10000000 --seed=0 --gradient_steps_per_step=1
python stream_ac_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=10000000 --seed=1 --gradient_steps_per_step=1
python stream_ac_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=10000000 --seed=2 --gradient_steps_per_step=1

python stream_ac_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=0 --gradient_steps_per_step=2
python stream_ac_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=1 --gradient_steps_per_step=2
python stream_ac_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=2 --gradient_steps_per_step=2

python stream_ac_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=0 --gradient_steps_per_step=4
python stream_ac_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=1 --gradient_steps_per_step=4
python stream_ac_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=2 --gradient_steps_per_step=4

python stream_ac_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=0 --gradient_steps_per_step=8
python stream_ac_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=1 --gradient_steps_per_step=8
python stream_ac_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=2 --gradient_steps_per_step=8

python stream_q_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=0 --gradient_steps_per_step=2
python stream_q_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=1 --gradient_steps_per_step=2
python stream_q_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=2 --gradient_steps_per_step=2

python stream_q_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=0 --gradient_steps_per_step=4
python stream_q_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=1 --gradient_steps_per_step=4
python stream_q_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=2 --gradient_steps_per_step=4

python stream_q_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=0 --gradient_steps_per_step=8
python stream_q_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=1 --gradient_steps_per_step=8
python stream_q_continuous.py --debug --env Humanoid-v4 --track=1 --total_steps=500000 --seed=2 --gradient_steps_per_step=8
