import pandas as pd

# Load merged file
df = pd.read_excel('Merged_Courses.xlsx')

# ------------------------------
# 1. DROP COMPLETELY EMPTY COLUMNS
# ------------------------------
df = df.dropna(axis=1, how='all')

# ------------------------------
# 2. DROP USELESS "Unnamed" COLUMNS
# ------------------------------
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# ------------------------------
# 3. DROP DUPLICATE COLUMNS
# ------------------------------
df = df.loc[:, ~df.columns.duplicated()]

# ------------------------------
# 4. OPTIONAL: DROP LOW-VALUE COLUMNS
# Auto-remove columns with >95% NaN values
# ------------------------------
threshold = 0.95  # drop columns where more than 95% rows are empty
df = df.loc[:, df.isna().mean() < threshold]

# ------------------------------
# 5. OPTIONAL: Keep only meaningful columns explicitly (recommended)
# You can customize this list based on your dataset.
# ------------------------------
meaningful_cols = [
    'department',
    'course_code',
    'course_title',
    'course_description',
    'credits',
    'prerequisites',
    'Course Instructor',
    'Instructor Name',
    'Instructor',
    'Slot Name',
    'Instructor Email',
    'Lecture Time',
    'Tutorial Time',
    'Practical Time',
    'Room',
    'Course ID',
    'norm_code',
    'final_code'
]

# Keep only the columns that exist
final_cols = [c for c in meaningful_cols if c in df.columns]

df_cleaned = df[final_cols]

# ------------------------------
# 6. SAVE OUTPUT
# ------------------------------
output_path = 'Merged_Courses_CLEANED.xlsx'
df_cleaned.to_excel(output_path, index=False)

output_path
