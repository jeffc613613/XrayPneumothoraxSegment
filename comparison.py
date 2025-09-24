import json
import matplotlib.pyplot as plt
import pathlib
import re
import numpy as np

experiment_dirs = [
    # r"D:\XrayPnxSegment\checkpoints\PSP_Net",
    # r"D:\XrayPnxSegment\checkpoints\FPN",
    # r"D:\XrayPnxSegment\checkpoints\DeepLabV3",
    r"X:\DeepLabv3_stage2\checkpoints\2509151510_overfitting"
]

file_map = {}
stage_lengths = {}

for exp_dir in experiment_dirs:
    exp_path = pathlib.Path(exp_dir)
    exp_name = exp_path.name
    for jf in exp_path.glob("*.json"):
        m = re.search(r"(stage\d+)", jf.stem.lower())
        if m:
            stage = m.group(1)
            file_map.setdefault(exp_name, {})[stage] = jf
            with open(jf, "r") as f:
                data = json.load(f)
            stage_lengths[stage] = len(data.get("val_dice", []))

stage_boundaries = []
offset = 0
for stage in sorted(stage_lengths.keys()):
    start = offset
    end = offset + stage_lengths[stage]
    stage_boundaries.append((stage, start, end))
    offset = end + 2

def add_label(ax, x, y, text, color, above=True, existing_labels=[]):
    """畫文字標籤，智能避讓"""
    y_offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    x_offset = 0.0
    if not above:
        y_offset *= -1

    tx, ty = x + x_offset, y + y_offset

    for lx, ly in existing_labels:
        if abs(tx - lx) < 100:
            if tx >= lx:
                tx += 2 
            else:
                tx -= 2 

    ax.text(tx, ty, text,
            fontsize=8 if not "Global" in text else 9,
            ha="center", va="bottom" if above else "top",
            color=color, weight="bold" if "Global" in text else "normal",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    return (tx, ty)

def plot_all_metrics():
    fig, axes = plt.subplots(2, 2, figsize=(14,10))

    metrics = [
        ("val_dice", "Validation Dice", "Val Dice"),
        ("val_iou", "Validation IoU", "Val IoU"),
        ("val_loss", "Validation Loss", "Val Loss"),
        ("val_accuracy", "Validation Accuracy", "Val Accuracy")
    ]

    for ax, (metric_key, title, ylabel) in zip(axes.flatten(), metrics):
        best_global_val = None
        best_global_x, best_global_y, best_global_exp = None, None, None
        labels_positions = []
        
        for exp_name, stages in file_map.items():
            x, y = [], []
            epoch_offset = 0
            for stage in sorted(stages.keys()):
                with open(stages[stage], "r") as f:
                    data = json.load(f)
                vals = data.get(metric_key, [])
                epochs = list(range(epoch_offset, epoch_offset + len(vals)))
                x.extend(epochs)
                y.extend(vals)
                epoch_offset += len(vals) + 2

            if not y:
                continue

            ax.plot(x, y, label=exp_name)

            # 每個實驗最佳點
            if metric_key == "val_loss":
                best_idx = int(np.argmin(y))
            else:
                best_idx = int(np.argmax(y))
            bx, by = x[best_idx], y[best_idx]

            ax.scatter(bx, by, color=ax.lines[-1].get_color(),
                       marker='o', s=60, zorder=5)
            labels_positions.append(
                add_label(ax, bx, by, f"Ep{bx}\n{by:.3f}",
                          color=ax.lines[-1].get_color(),
                          above=False, existing_labels=labels_positions)
            )

            # 更新全域最佳
            if best_global_val is None:
                best_global_val = by
                best_global_x, best_global_y, best_global_exp = bx, by, exp_name
            else:
                if (metric_key == "val_loss" and by < best_global_val) or \
                   (metric_key != "val_loss" and by > best_global_val):
                    best_global_val, best_global_x, best_global_y, best_global_exp = by, bx, by, exp_name

        # 全域最佳 (星形)
        # if best_global_x is not None:
        #     ax.scatter(best_global_x, best_global_y,
        #                color="red", marker="*", s=200, zorder=6)
        #     labels_positions.append(
        #         add_label(ax, best_global_x, best_global_y,
        #                   f"Global {best_global_exp}\nEp{best_global_x}, {best_global_val:.3f}",
        #                   color="red", above=True, existing_labels=labels_positions)
        #     )

        # Stage 分界虛線 + 標籤
        for i, (stage, start, end) in enumerate(stage_boundaries):
            ax.axvline(x=end, color="gray", linestyle="--", alpha=0.35)
            ax.text((start+end)//2,
                ax.get_ylim()[1] * 0.5,   # 改成靠近圖內上方
                stage.upper(),
                ha="center", va="top", fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.show()

plot_all_metrics()
