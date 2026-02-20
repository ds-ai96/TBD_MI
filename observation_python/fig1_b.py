import matplotlib.pyplot as plt

def plot_performance_curve(data1, data2, data3):
    """
    data1, data2, data3: {num_images: performance}
    각 data에 대해 line을 그려서 하나의 그림에 3개의 line을 표시
    """

    colors = ['red', 'green', 'blue']
    labels = ['DeepInversion', 'SMI', 'PAINT (proposed)']
    datasets = [data1, data2, data3]

    plt.figure(figsize=(4, 4))

    all_x = []
    for data_dict, color, label in zip(datasets, colors, labels):
        if labels in ["DeepInversion", "PAINT (proposed)"]:
            continue
        items = sorted(data_dict.items())
        num_images = [k for k, _ in items]
        performance = [v for _, v in items]
        all_x.extend(num_images)

        plt.plot(
            num_images,
            performance,
            marker='o',
            markersize=12,
            markerfacecolor='none',
            markeredgewidth=2,
            linewidth=2,
            color=color,
            label=label
        )

    plt.xlabel("Elapsed Time", fontsize=18)
    plt.ylabel("Performance", fontsize=18)

    # x-axis: 400 단위
    max_x = max(all_x)
    plt.xlim(-300, max_x * 1.1)
    plt.xticks(range(0, int(max_x * 1.1) + 1, 400))
    plt.tick_params(axis='both', labelsize=16)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("./observation_python/figures/fig1_b.jpg", dpi=600, format="jpg", bbox_inches="tight")


# -------------------------
# Example usage
# -------------------------
dmi = {
    # 400: 70.95,
    # 1200: 70.76,
    # 1600: 75.39,
}

smi = {
    400: 52.67,
    800: 55.35,
    1600: 59.37,
}

paint = {
#     400: 70.95,
#     1200: 70.76,
#     1600: 75.39,
}

plot_performance_curve(dmi, smi, paint)