import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = r"runs\detect\hazards_yolov8n\results.csv"
OUT_PNG = r"runs\detect\hazards_yolov8n\results.png"
ARTIFACT_PNG = r"C:\Users\esakk\.gemini\antigravity-ide\brain\6bcd2eec-6be0-4ee0-9f8f-3d1601a88f5e\results.png"

def plot_results():
    if not os.path.exists(RESULTS_CSV):
        print(f"[!] Error: {RESULTS_CSV} not found")
        return

    df = pd.read_csv(RESULTS_CSV)
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    print("Columns found in results.csv:")
    print(df.columns.tolist())

    print(f"Data rows count: {len(df)}")
    print("Head:")
    print(df.head(2))

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), tight_layout=True)
    fig.suptitle("YOLOv8n Training & Validation Metrics (100 Epochs)", fontsize=16, fontweight='bold')

    plots = [
        ("train/box_loss", "Train Box Loss", "blue"),
        ("train/cls_loss", "Train Class Loss", "orange"),
        ("train/dfl_loss", "Train DFL Loss", "purple"),
        ("metrics/precision(B)", "Precision (B)", "green"),
        ("metrics/recall(B)", "Recall (B)", "teal"),
        ("val/box_loss", "Val Box Loss", "red"),
        ("val/cls_loss", "Val Class Loss", "magenta"),
        ("val/dfl_loss", "Val DFL Loss", "brown"),
        ("metrics/mAP50(B)", "mAP@50 (B)", "darkblue"),
        ("metrics/mAP50-95(B)", "mAP@50-95 (B)", "darkgreen")
    ]

    for idx, (col, title, color) in enumerate(plots):
        r, c = divmod(idx, 5)
        ax = axes[r, c]
        if col in df.columns:
            epochs = df["epoch"] if "epoch" in df.columns else df.index + 1
            ax.plot(epochs, df[col], color=color, linewidth=2, label=title)
            ax.set_title(title, fontsize=12, fontweight='semibold')
            ax.set_xlabel("Epoch", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.6)
            # Annotate final value
            final_val = df[col].iloc[-1]
            ax.annotate(f"{final_val:.4f}", xy=(epochs.iloc[-1], final_val),
                        xytext=(-30, 10), textcoords="offset points",
                        fontweight="bold", color=color,
                        arrowprops=dict(arrowstyle="->", color=color))
        else:
            ax.set_title(f"{title} (N/A)", fontsize=12)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Successfully generated high-res custom {OUT_PNG}")

    # Copy to artifact directory
    import shutil
    shutil.copy2(OUT_PNG, ARTIFACT_PNG)
    print(f"[+] Copied to artifact dir: {ARTIFACT_PNG}")

if __name__ == "__main__":
    plot_results()
