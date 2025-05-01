import numpy as np
import matplotlib.pyplot as plt

def plot_results(save_path: str = None) -> None:
    """Plots hardcoded results from the VLM-ITL experiments.
    The results include the baseline, AL, and VLM results.
    The x-axis is the percentage of the dataset used for training,
    and the y-axis is the mean IoU score.
    The plot includes a horizontal line for the baseline result,
    and lineplots for the AL and VLM results.

    Args:
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    baseline_result = np.float16(0.740)
    al_results_rand = np.array([0.4187, 0.6264, 0.6927, 0.716, 0.774])
    al_results_lc = np.array([0.5626, 0.7053, 0.7533, 0.7606, 0.7784])
    vlm_results_rand = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    vlm_results_lc = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    x_axis = np.arange(0.2, 1.0, 0.2)
    x_axis = np.concatenate((x_axis, [1.0]))
    plt.figure(figsize=(15, 9))
    plt.plot(x_axis, al_results_rand, marker='o', label='AL (Random Sampling)')
    plt.plot(x_axis, al_results_lc, marker='o', label='AL (LC Sampling)')
    plt.plot(x_axis, vlm_results_rand, marker='o', label='VLM (Random Sampling)')
    plt.plot(x_axis, vlm_results_lc, marker='o', label='VLM (LC + VLM Sampling)')
    plt.axhline(baseline_result, color='r', linestyle='--', label='Baseline (Full Supervision)')
    plt.xticks(x_axis)
    plt.xlabel('Percentage of Dataset Used for Training')
    plt.ylabel('Mean IoU Score')
    plt.title('VLM-ITL Results')
    plt.legend()
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
    return None

if __name__ == "__main__":
    plot_results(save_path='vlm_itl_results.png')