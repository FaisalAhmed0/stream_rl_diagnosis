import numpy as np
import pickle, os
import matplotlib.pyplot as plt
import glob

def avg_return_curve(x, y, stride, total_steps):
    """
    Author: Rupam Mahmood (armahmood@ualberta.ca)
    :param x: A list of list of termination steps for each episode. len(x) == total number of runs
    :param y: A list of list of episodic return. len(y) == total number of runs
    :param stride: The timestep interval between two aggregate datapoints to be calculated
    :param total_steps: The total number of time steps to be considered
    :return: time steps for calculated data points, average returns for each data points, std-errs
    """
    assert len(x) == len(y)
    num_runs = len(x)
    avg_ret = np.zeros(total_steps // stride)
    stderr_ret = np.zeros(total_steps // stride)
    steps = np.arange(stride, total_steps + stride, stride)
    for i in range(0, total_steps // stride):
        rets = []
        avg_rets_per_run = []
        for run in range(num_runs):
            xa = np.array(x[run])
            ya = np.array(y[run])
            rets.append(ya[np.logical_and(i * stride < xa, xa <= (i + 1) * stride)].tolist())
            avg_rets_per_run.append(np.mean(rets[-1]))
        avg_ret[i] = np.mean(avg_rets_per_run)
        stderr_ret[i] = np.std(avg_rets_per_run) / np.sqrt(num_runs)
    return steps, avg_ret, stderr_ret

def parse_h_nl(dir_name):
    h, nl = 0, 0
    for part in dir_name.split('_'):
        if part.startswith('h') and part[1:].isdigit():
            h = int(part[1:])
        elif part.startswith('nl') and part[2:].isdigit():
            nl = int(part[2:])
    return h, nl

def get_env_and_algo_names(dir_path):
    pkl_files = [f for f in os.listdir(dir_path) if f.endswith(".pkl")]
    with open(os.path.join(dir_path, pkl_files[0]), "rb") as f:
        episodic_returns, termination_time_steps, env_name = pickle.load(f)
    parts = os.path.basename(dir_path).split('_')
    algo_name = parts[2].upper()
    return env_name, algo_name

def plot_dirs(dirs, int_space, total_steps, title, out_png):
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(dirs)))

    env_name = ""
    for idx, dir_path in enumerate(dirs):
        all_termination_time_steps, all_episodic_returns = [], []
        pkl_files = [f for f in os.listdir(dir_path) if f.endswith(".pkl")]
        for file in pkl_files:
            with open(os.path.join(dir_path, file), "rb") as f:
                episodic_returns, termination_time_steps, env_name = pickle.load(f)
                all_termination_time_steps.append(termination_time_steps)
                all_episodic_returns.append(episodic_returns)

        h, nl = parse_h_nl(os.path.basename(dir_path))
        steps, avg_ret, stderr_ret = avg_return_curve(
            all_termination_time_steps, all_episodic_returns, int_space, total_steps
        )

        label = f"h={h}, nl={nl}"
        color = colors[idx]
        plt.fill_between(steps, avg_ret - stderr_ret, avg_ret + stderr_ret, color=color, alpha=0.10)
        plt.plot(steps, avg_ret, linewidth=3.5, color=color, label=label)

    plt.xlabel("Time Step", fontsize=20)
    plt.ylabel("Average Episodic Return", fontsize=20)
    plt.title(title, fontsize=22)
    plt.legend(loc="best", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main(data_dir, fixed_h, fixed_nl, int_space, total_steps):
    all_dirs = sorted([d for d in glob.glob(data_dir) if os.path.isdir(d)])
    env_name, algo_name = get_env_and_algo_names(all_dirs[0])

    meta = sorted([(parse_h_nl(os.path.basename(d)), d) for d in all_dirs])

    # fixed width and vary depth
    depth_sweep = [d for (h, nl), d in meta if h == fixed_h]
    plot_dirs(
        depth_sweep,
        int_space,
        total_steps,
        title=f"Stream {algo_name}(0.8) on {env_name}: fixed width h={fixed_h} and varying depth",
        out_png=f"{env_name}_fixh{fixed_h}_vary_nl.png",
    )

    # fixed depth and vary width
    width_sweep = [d for (h, nl), d in meta if nl == fixed_nl]
    plot_dirs(
        width_sweep,
        int_space,
        total_steps,
        title=f"Stream {algo_name}(0.8) on {env_name}: fixed depth nl={fixed_nl} and varying width",
        out_png=f"{env_name}_fixnl{fixed_nl}_vary_h.png",
    )

if __name__ == '__main__':
    # for instance: python plot.py --data_dir "data_stream_ac_HalfCheetah-v4_*" --hidden_size 128 --num_layers 2
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=32)
    parser.add_argument('--int_space', type=int, default=50_000)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    args = parser.parse_args()
    main(args.data_dir, args.hidden_size, args.num_layers, args.int_space, args.total_steps)