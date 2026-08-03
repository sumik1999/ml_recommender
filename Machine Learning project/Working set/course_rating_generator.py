import pandas as pd
import numpy as np
import random

# -----------------------------
# CONFIGURATION
# -----------------------------
EXCEL_PATH = "trial/courses_dataset_new-2.xlsx"         # input file
OUTPUT_PATH = "trial/courses_populated.xlsx"

CAPACITY_OPTIONS = [30, 40, 50, 60, 70]

# -----------------------------
# ENROLLMENT GENERATOR
# -----------------------------
def generate_enrollment(rating):
    """
    Generate students_enrolled_tilldate based on rating.
    Higher rating → higher enrollment.
    """
    if rating >= 4.5:
        return random.randint(300, 500)
    elif rating >= 4.0:
        return random.randint(200, 300)
    elif rating >= 3.0:
        return random.randint(100, 200)
    elif rating >= 2.0:
        return random.randint(50, 100)
    else:
        return random.randint(10, 50)

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def populate_excel(excel_path, output_path):

    # Load Excel with all existing columns preserved
    df = pd.read_excel(excel_path, engine="openpyxl")

    # ----- ADD NEW COLUMNS (does NOT overwrite existing columns) -----

    # 1. capacity
    df["capacity"] = np.random.choice(CAPACITY_OPTIONS, size=len(df))

    # 2. course_rating
    df["course_rating"] = np.round(
        np.random.uniform(1.0, 5.0, size=len(df)),
        1
    )

    # 3. students_enrolled_tilldate (based on rating)
    df["students_enrolled_tilldate"] = df["course_rating"].apply(generate_enrollment)

    # Save updated file
    df.to_excel(output_path, index=False)
    print(f"\nSUCCESS: File saved → {output_path}")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    populate_excel(EXCEL_PATH, OUTPUT_PATH)
