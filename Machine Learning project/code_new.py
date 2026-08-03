# recommender.py
# Requirements: pandas, scikit-learn, numpy, python-Levenshtein (optional)
# pip install pandas scikit-learn numpy

import json
import re
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONFIG ----------
COURSE_CSV_PATH = "/mnt/data/final_human_written_course_dataset_MASTER.csv"
# If you have a mapping.json of prefixes -> departments, optionally load it:
MAPPING_JSON_PATH = "/mnt/data/mapping.json"  # optional, used if present

# ---------- UTILITIES ----------
def safe_text(x):
    if pd.isna(x): return ""
    if isinstance(x, (list, tuple)): return " ".join(map(str, x))
    return str(x)

def normalize_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def blob_for_course(row: pd.Series) -> str:
    # Combine title, description, competencies (if stringified list) and department
    parts = [
        safe_text(row.get("Course_title","")),
        safe_text(row.get("Course_description","")),
        safe_text(row.get("Course_competencies","")),
        safe_text(row.get("Course_department","")),
    ]
    blob = " . ".join([normalize_whitespace(p) for p in parts if p])
    return blob

# ---------- LOAD DATA ----------
def load_courses(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure necessary columns exist
    for c in ["Course_id","Course_title","Course_description","Course_competencies",
              "Course_department","Course_slot","Course_credits","Course_prerequisites"]:
        if c not in df.columns:
            df[c] = ""
    # Build combined text blob
    df["__text_blob__"] = df.apply(blob_for_course, axis=1)
    # Normalize competencies column: if it's a stringified list, we keep it
    df["Course_competencies"] = df["Course_competencies"].fillna("").astype(str)
    # Course_credits numeric
    def parse_credits(x):
        try:
            return float(x)
        except:
            # if string like "3.0-0.0-2.0" -> sum first numeric(s) or extract final credit?
            # We'll fallback to 3.0 if unavailable
            m = re.findall(r"[\d.]+", str(x))
            return float(m[-1]) if m else 0.0
    df["Course_credits_num"] = df["Course_credits"].apply(parse_credits)
    return df

# ---------- MODEL BUILD ----------
class CourseRecommender:
    def __init__(self, course_df: pd.DataFrame, vectorizer: Optional[TfidfVectorizer]=None):
        self.df = course_df.reset_index(drop=True)
        self.vectorizer = vectorizer or TfidfVectorizer(
            max_features=20000,
            ngram_range=(1,2),
            stop_words="english"
        )
        # Fit vectorizer on course blobs
        self.course_texts = self.df["__text_blob__"].fillna("").astype(str).tolist()
        self.tfidf = self.vectorizer.fit_transform(self.course_texts)
        # Precompute a map from course_id -> row index
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.df["Course_id"].astype(str))}
    
    def build_user_vector(self, user_profile: Dict[str,Any]) -> np.ndarray:
        # Build a textual query from user fields: interests + competencies + career interest + previous courses
        pieces = []
        # user_course_interest may be list or string
        pieces.append(" ".join(user_profile.get("User_course_interest", []) if isinstance(user_profile.get("User_course_interest", []), list)
                               else [user_profile.get("User_course_interest","")]))
        pieces.append(" ".join(user_profile.get("User_competencies", []) if isinstance(user_profile.get("User_competencies", []), list)
                               else [user_profile.get("User_competencies","")]))
        pieces.append(user_profile.get("User_career_interest",""))
        # include previous courses titles/ids (helps match pre-reqs & topic familiarity)
        prev = user_profile.get("User_previous_courses", [])
        if isinstance(prev, list):
            pieces.append(" ".join(prev))
        else:
            pieces.append(str(prev))
        query = normalize_whitespace(" . ".join([p for p in pieces if p]))
        # transform via vectorizer (same vocabulary)
        user_vec = self.vectorizer.transform([query])
        return user_vec

    def recommend(self,
                  user_profile: Dict[str,Any],
                  top_n: int = 10,
                  min_credits: Optional[float] = None,
                  max_credits: Optional[float] = None,
                  department_whitelist: Optional[List[str]] = None,
                  exclude_slots: Optional[List[str]] = None,
                  prereq_penalty: float = 0.25
                 ) -> pd.DataFrame:
        """
        Returns top-N recommended courses as a DataFrame with score breakdown.
        - prereq_penalty: fraction of similarity to subtract if user lacks prereqs
        """
        user_vec = self.build_user_vector(user_profile)
        sims = cosine_similarity(user_vec, self.tfidf).flatten()  # similarity scores
        scores = sims.copy()

        # Filtering: slots
        if exclude_slots:
            mask_slots = self.df["Course_slot"].isin(exclude_slots)
            scores[mask_slots.values] = -1.0  # never recommend

        # Filtering: department
        if department_whitelist:
            dept_mask = ~self.df["Course_department"].isin(department_whitelist)
            scores[dept_mask.values] *= 0.5  # downweight others (instead of removal), can change to removal

        # Filter by credit bounds
        if min_credits is not None:
            mask = self.df["Course_credits_num"] < min_credits
            scores[mask.values] = -1.0
        if max_credits is not None:
            mask = self.df["Course_credits_num"] > max_credits
            scores[mask.values] = -1.0

        # Prerequisite handling: if Course_prerequisites contains course ids, penalize
        user_prev = {str(x).strip().upper() for x in (user_profile.get("User_previous_courses") or [])}
        for i, prereq in enumerate(self.df["Course_prerequisites"].astype(str).fillna("")):
            if not prereq or prereq.strip() in ("[]", "nan"):
                continue
            # we expect prereq to be either a stringified list like "['MATH101','CS101']" or "MATH101"
            # extract tokens that look like course ids: capital letters + digits
            tokens = re.findall(r"[A-Z]{2,5}[\s\-]?\d{2,5}", prereq.upper())
            tokens = {t.replace(" ", "").replace("-", "") for t in tokens}
            if tokens and not tokens.intersection(user_prev):
                # user lacks prerequisites -> penalize this course
                scores[i] = scores[i] * (1.0 - prereq_penalty)

        # Build output DataFrame
        out = self.df.copy()
        out["raw_similarity"] = sims
        out["score"] = scores
        out["reason_blob"] = out["__text_blob__"].str.slice(0,300)  # short preview
        # Sort and return top_n
        out = out.sort_values("score", ascending=False)
        # Remove negative-scored entries (filtered)
        out = out[out["score"] > 0].head(top_n)
        # Return only useful columns
        return out[["Course_id","Course_title","Course_department","Course_credits_num",
                    "Course_slot","Course_instructor","score","raw_similarity","reason_blob"]]

# ---------- EXAMPLE USAGE ----------
def demo():
    df = load_courses(COURSE_CSV_PATH)
    rec = CourseRecommender(df)

    # Sample user profile (structure you requested)
    user_profile = {
        "User_id": "U1001",
        "User_program": "Btech",
        "User_department": "CSE",
        "User_branch": "COL",  # branch/course code prefix
        "User_name": "Anita",
        "User_age": 20,
        "user_year_of_study": 2,
        "User_previous_courses": ["COL1000", "COL106", "COL216"],  # course ids
        "User_course_interest": ["Databases", "Distributed Systems", "Machine Learning"],
        "User_career_interest": "Data Science",
        "User_competencies": ["Python", "SQL", "Linear Algebra"],
        "User_grades": {"COL1000": "A", "COL106": "A-", "COL216": "B+"},
        "User_occupiedslots": ["A","B"],  # sample slots user cannot take
        "User_credits_requirement": 12.0,
    }

    recommendations = rec.recommend(
        user_profile=user_profile,
        top_n=12,
        department_whitelist=None,
        exclude_slots=user_profile.get("User_occupiedslots", []),
        min_credits=None,
        max_credits=None,
        prereq_penalty=0.25
    )

    print("Top recommendations for", user_profile["User_name"])
    print(recommendations.to_string(index=False))

if __name__ == "__main__":
    demo()
