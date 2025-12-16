# Streaming Deep Reinforcement Learning

# RL class project 

## Rerunning experiments

Commands to reproduce the full set of runs (seeds 0, 1, 2).

### First batch

```bash
# CartPole-v1 (Stream Q(λ))
for h in 32 128 512; do
  for nl in 1 2 4; do
    for seed in 0 1 2; do
      python stream_q.py \
        --env_name CartPole-v1 \
        --seed "${seed}" \
        --hidden_size "${h}" \
        --num_layers "${nl}" \
        --total_steps 500000
    done
  done
done

# HalfCheetah-v4 (Stream AC(λ))
for h in 64 128 256; do
  for nl in 1 2 4; do
    for seed in 0 1 2; do
      python stream_ac_continuous.py \
        --env_name HalfCheetah-v4 \
        --seed "${seed}" \
        --hidden_size "${h}" \
        --num_layers "${nl}" \
        --total_steps 500000
    done
  done
done
```

### Second Batch

```bash
# Hopper-v4 and Humanoid-v4 (Stream AC(λ))
for ENV in Hopper-v4 Humanoid-v4; do
  for H in 64 256; do
    for NL in 2 4 8; do
      for SEED in 0 1 2; do
        python stream_ac_continuous.py \
          --env_name "$ENV" \
          --hidden_size "$H" \
          --num_layers "$NL" \
          --seed "$SEED" \
          --total_steps 2500000
      done
    done
  done
done
```