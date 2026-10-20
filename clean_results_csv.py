import os

RESULTS_CSV = r"runs\detect\hazards_yolov8n\results.csv"
CLEAN_CSV = r"runs\detect\hazards_yolov8n\results.csv"

def clean_csv():
    with open(RESULTS_CSV, "r") as f:
        lines = f.readlines()

    header = lines[0].strip().split(",")
    num_cols = len(header)
    print(f"Header has {num_cols} columns: {header}")

    cleaned_lines = [lines[0].strip()]

    for idx, line in enumerate(lines[1:], start=2):
        parts = line.strip().split(",")
        if len(parts) >= num_cols:
            # Take only the first 15 columns
            clean_part = parts[:num_cols]
            cleaned_lines.append(",".join(clean_part))
        else:
            print(f"[!] Warning: Line {idx} has fewer than {num_cols} columns ({len(parts)}). Skipping.")

    with open(CLEAN_CSV, "w") as f:
        f.write("\n".join(cleaned_lines) + "\n")

    print(f"[+] Successfully cleaned {RESULTS_CSV}. Total valid rows: {len(cleaned_lines)-1}")

if __name__ == "__main__":
    clean_csv()
