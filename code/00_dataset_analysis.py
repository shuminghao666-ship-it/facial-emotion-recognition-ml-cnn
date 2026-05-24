# ============================================================
# Pipeline Step 00:
# Dataset Analysis for Facial Emotion Recognition
#
# Purpose:
# 1. Count training/evaluation samples for each emotion class.
# 2. Calculate class percentages and class imbalance ratio.
# 3. Check unreadable or corrupted images.
# 4. Save distribution tables and figures for the project report.
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from project_config import (
    CLASSES,
    CLASS_FOLDER_NAMES,
    IMAGE_EXTENSIONS,
    RESULT_DIR,
    TEST_DIR,
    TRAIN_DIR,
)


OUTPUT_DIR = RESULT_DIR / "dataset_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = set(IMAGE_EXTENSIONS)


def find_class_dir(split_dir, class_name):
    for folder_name in CLASS_FOLDER_NAMES[class_name]:
        candidate = split_dir / folder_name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def list_image_files(class_dir):
    if class_dir is None:
        return []
    return sorted(
        file
        for file in class_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def count_split(split_name, split_dir):
    rows = []
    for class_name in CLASSES:
        class_dir = find_class_dir(split_dir, class_name)
        image_files = list_image_files(class_dir)
        rows.append(
            {
                "Split": split_name,
                "Class": class_name,
                "Folder": str(class_dir) if class_dir else "",
                "Count": len(image_files),
            }
        )
    return rows


def build_distribution_table():
    rows = []
    rows.extend(count_split("Train", TRAIN_DIR))
    rows.extend(count_split("Eval", TEST_DIR))

    df = pd.DataFrame(rows)
    total_by_split = df.groupby("Split")["Count"].transform("sum")
    df["Percentage"] = (df["Count"] / total_by_split * 100).round(2)
    df["Total_In_Split"] = total_by_split

    total_rows = []
    for class_name in CLASSES:
        train_count = int(
            df[(df["Split"] == "Train") & (df["Class"] == class_name)]["Count"].iloc[0]
        )
        test_count = int(
            df[(df["Split"] == "Eval") & (df["Class"] == class_name)]["Count"].iloc[0]
        )
        total_rows.append(
            {
                "Class": class_name,
                "Train_Count": train_count,
                "Eval_Count": test_count,
                "Total_Count": train_count + test_count,
            }
        )

    summary_df = pd.DataFrame(total_rows)
    summary_total = summary_df["Total_Count"].sum()
    summary_df["Total_Percentage"] = (
        summary_df["Total_Count"] / summary_total * 100
    ).round(2)

    return df, summary_df


def check_images():
    problem_rows = []

    for split_name, split_dir in [("Train", TRAIN_DIR), ("Eval", TEST_DIR)]:
        for class_name in CLASSES:
            class_dir = find_class_dir(split_dir, class_name)
            for image_path in list_image_files(class_dir):
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as exc:
                    problem_rows.append(
                        {
                            "Split": split_name,
                            "Class": class_name,
                            "Path": str(image_path),
                            "Error": str(exc),
                        }
                    )

    return pd.DataFrame(problem_rows)


def plot_class_distribution(summary_df):
    x = range(len(summary_df))
    width = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(
        [i - width / 2 for i in x],
        summary_df["Train_Count"],
        width=width,
        label="Train",
    )
    plt.bar(
        [i + width / 2 for i in x],
        summary_df["Eval_Count"],
        width=width,
        label="Held-out Evaluation",
    )
    plt.xticks(list(x), summary_df["Class"], rotation=25)
    plt.ylabel("Number of Images")
    plt.title("Class Distribution in Training and Held-out Evaluation Splits")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution_train_test.png", dpi=300)
    plt.close()


def plot_total_percentage(summary_df):
    plt.figure(figsize=(8, 8))
    plt.pie(
        summary_df["Total_Count"],
        labels=summary_df["Class"],
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Overall Class Percentage")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overall_class_percentage.png", dpi=300)
    plt.close()


def write_text_summary(distribution_df, summary_df, problem_df):
    train_total = int(distribution_df[distribution_df["Split"] == "Train"]["Count"].sum())
    test_total = int(distribution_df[distribution_df["Split"] == "Eval"]["Count"].sum())
    overall_total = train_total + test_total

    max_row = summary_df.loc[summary_df["Total_Count"].idxmax()]
    min_row = summary_df.loc[summary_df["Total_Count"].idxmin()]
    imbalance_ratio = max_row["Total_Count"] / max(1, min_row["Total_Count"])

    lines = [
        "Dataset Analysis Summary",
        "=" * 60,
        "",
        f"Training images: {train_total}",
        f"Held-out evaluation images: {test_total}",
        f"Total images: {overall_total}",
        "",
        "Class distribution:",
        summary_df.to_string(index=False),
        "",
        "Class imbalance:",
        f"Most frequent class: {max_row['Class']} ({int(max_row['Total_Count'])} images)",
        f"Least frequent class: {min_row['Class']} ({int(min_row['Total_Count'])} images)",
        f"Imbalance ratio: {imbalance_ratio:.2f}:1",
        "",
        f"Unreadable or corrupted images: {len(problem_df)}",
    ]

    with open(OUTPUT_DIR / "dataset_analysis_summary.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Training directory does not exist:\n{TRAIN_DIR}")
    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Held-out evaluation directory does not exist:\n{TEST_DIR}"
        )

    distribution_df, summary_df = build_distribution_table()
    problem_df = check_images()

    distribution_df.to_csv(
        OUTPUT_DIR / "split_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_df.to_csv(
        OUTPUT_DIR / "overall_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    problem_df.to_csv(
        OUTPUT_DIR / "unreadable_images.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_class_distribution(summary_df)
    plot_total_percentage(summary_df)
    write_text_summary(distribution_df, summary_df, problem_df)

    print("Dataset analysis completed.")
    print(f"Results saved to:\n{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
