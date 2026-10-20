import pandas as pd
import json

CSV_PATH = r"F:\project\Nav and fusion\runs\detect\hazards_threshold_experiment\grid_search_summary.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 80)
print("  TOP CONFIGURATIONS BY HIGHEST RECALL (OVERALL)")
print("=" * 80)
print(df.sort_values(by="recall", ascending=False)[["conf", "iou", "precision", "recall", "f1", "map50", "fire_r", "flood_r", "debris_r"]].head(8).to_string(index=False))

print("\n" + "=" * 80)
print("  TOP CONFIGURATIONS BY HIGHEST F1 SCORE (OVERALL)")
print("=" * 80)
print(df.sort_values(by="f1", ascending=False)[["conf", "iou", "precision", "recall", "f1", "map50", "fire_r", "flood_r", "debris_r"]].head(8).to_string(index=False))

print("\n" + "=" * 80)
print("  CONFIDENCE THRESHOLD COMPARISON (FIXED IOU = 0.50)")
print("=" * 80)
df_iou50 = df[df["iou"] == 0.50]
print(df_iou50[["conf", "iou", "precision", "recall", "f1", "map50", "fire_r", "flood_r", "debris_r"]].to_string(index=False))

print("\n" + "=" * 80)
print("  FIRE / FLOOD / DEBRIS METRICS AT CONF=0.15 vs CONF=0.25 vs CONF=0.30 (IOU=0.50)")
print("=" * 80)
for c in [0.15, 0.20, 0.25, 0.30, 0.35]:
    r = df[(df["conf"] == c) & (df["iou"] == 0.50)].iloc[0]
    print(f"Conf={c:.2f}, IoU=0.50 -> Overall P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}, mAP50={r['map50']:.4f}")
    print(f"   Fire   : P={r['fire_p']:.4f}, R={r['fire_r']:.4f}, F1={r['fire_f1']:.4f}, mAP50={r['fire_map50']:.4f}")
    print(f"   Flood  : P={r['flood_p']:.4f}, R={r['flood_r']:.4f}, F1={r['flood_f1']:.4f}, mAP50={r['flood_map50']:.4f}")
    print(f"   Debris : P={r['debris_p']:.4f}, R={r['debris_r']:.4f}, F1={r['debris_f1']:.4f}, mAP50={r['debris_map50']:.4f}\n")
