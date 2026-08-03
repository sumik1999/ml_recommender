import pandas as pd
import re
from rapidfuzz import process, fuzz

# -----------------------------------------------------------
# STEP 1 — NORMALIZE COURSE CODES
# -----------------------------------------------------------

def normalize_code(code):
    """Normalize course codes by removing spaces and uppercasing."""
    if pd.isna(code):
        return None
    return str(code).strip().upper().replace(" ", "")


# -----------------------------------------------------------
# STEP 2 — REGEX TO EXTRACT COURSE IDs FROM COURSE NAME COLUMN
# -----------------------------------------------------------

course_id_regex = re.compile(r"\b([A-Za-z]{2,3}[LVDP]?\s*\d{2,4}[A-Z]?)\b")

def extract_course_id(name):
    """Extract the first valid course ID from 'Course Name' column."""
    if pd.isna(name):
        return None
    
    match = course_id_regex.search(str(name))
    if match:
        raw_code = match.group(1)
        return normalize_code(raw_code)
    
    return None


# -----------------------------------------------------------
# STEP 3 — FUZZY MATCHING FALLBACK
# -----------------------------------------------------------

def fuzzy_match(code, choices):
    """Fuzzy match a course code against available Excel codes."""
    if code is None:
        return None
    match, score, idx = process.extractOne(code, choices, scorer=fuzz.WRatio)
    return match if score >= 85 else None


# -----------------------------------------------------------
# STEP 4 — LOAD INPUT FILES
# -----------------------------------------------------------

json_path = "parsed_courses.json"
excel_path = "Courses_Offered.xlsx"

print("Loading files...")

json_df = pd.read_json(json_path)
excel_df = pd.read_excel(excel_path,skiprows=4)
print(excel_df.columns.tolist())


# -----------------------------------------------------------
# STEP 5 — EXTRACT COURSE ID FROM EXCEL COURSE NAME COLUMN
# -----------------------------------------------------------

print("Extracting Course IDs from Excel...")

excel_df["Course ID"] = excel_df["Course Name"].apply(extract_course_id)

# Normalize it again just to be safe
excel_df["norm_code"] = excel_df["Course ID"].apply(normalize_code)


# -----------------------------------------------------------
# STEP 6 — NORMALIZE JSON COURSE CODES
# -----------------------------------------------------------

json_df["norm_code"] = json_df["course_code"].apply(normalize_code)


# -----------------------------------------------------------
# STEP 7 — FUZZY MATCHING JSON CODES TO EXCEL CODES
# -----------------------------------------------------------

excel_codes = excel_df["norm_code"].dropna().unique().tolist()

print("Running fuzzy matching on JSON codes...")

json_df["fuzzy_match"] = json_df["norm_code"].apply(
    lambda code: fuzzy_match(code, excel_codes)
)

# Final code for merge = fuzzy match if exists, else normalized code
json_df["final_code"] = json_df.apply(
    lambda r: r["fuzzy_match"] if pd.notna(r["fuzzy_match"]) else r["norm_code"],
    axis=1
)


# -----------------------------------------------------------
# STEP 8 — MERGE JSON WITH EXCEL USING final_code
# -----------------------------------------------------------

print("Merging JSON data with Excel offerings...")

merged = json_df.merge(
    excel_df,
    left_on="final_code",
    right_on="norm_code",
    how="left",
    suffixes=("_parsed", "_offered")
)


# -----------------------------------------------------------
# STEP 9 — FLAG COURSES NOT FOUND IN EXCEL
# -----------------------------------------------------------

not_found = merged[merged["Course ID"].isna()].copy()


# -----------------------------------------------------------
# STEP 10 — SAVE FINAL OUTPUT FILES
# -----------------------------------------------------------

output_merged = "Merged_Courses.xlsx"
output_not_found = "Courses_Not_Found.xlsx"

merged.to_excel(output_merged, index=False)
not_found.to_excel(output_not_found, index=False)

print("\n✔ PROCESS COMPLETED SUCCESSFULLY!")
print(f"➡ Merged file saved to: {output_merged}")
print(f"➡ Unmatched courses saved to: {output_not_found}")
