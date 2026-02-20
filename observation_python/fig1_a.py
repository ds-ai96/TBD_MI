import matplotlib.pyplot as plt

def plot_performance_curve(data_dict):
    """
    data_dict: {num_images: performance}
    """

    # Sort by number of images (x-axis 정렬 보장)
    items = sorted(data_dict.items())
    num_images = [k for k, _ in items]
    performance = [v for _, v in items]

    plt.figure(figsize=(4, 4))

    plt.plot(
        num_images,
        performance,
        marker='o',
        markersize=12,
        markerfacecolor='none',
        markeredgewidth=2,
        linewidth=2,
        color='blue'
    )

    plt.xlabel("Number of images", fontsize=18)
    plt.ylabel("Performance", fontsize=18)
    # plt.title("Sparse Model Inversion: Performance vs. # Images", fontsize=13)

    # x-axis: 1000 단위
    max_x = max(num_images)
    plt.xlim(-300, max_x * 1.1)
    plt.xticks(range(0, int(max_x * 1.1) + 1, 1000))
    plt.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.savefig("fig1_a.jpg", dpi=600, format="jpg", bbox_inches="tight")


# -------------------------
# Example usage
# -------------------------
data = {
    8: 24.35,
    16: 41.05,
    32: 51.01,
    64: 67.67,
    128: 81.51,
    512: 82.37,
    2048: 84.63,
    4096: 86.52,
}

plot_performance_curve(data)