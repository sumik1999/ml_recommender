# recommender_semantic.py
# Requires: pandas, scikit-learn, numpy, sentence-transformers
# pip install pandas scikit-learn numpy sentence-transformers

import json
import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import os
import pickle

# ---------- CONFIG ----------
COURSE_CSV_PATH = "/generated_course_dataset.csv"
MAPPING_JSON_PATH = "/mapping.json"  # optional if you use it
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"        # chosen option A (fast & accurate)
EMBEDDINGS_CACHE = "/course_embeddings_sbert.pkl"  # cache embeddings for speed

# Hybrid weights (tuneable)
ALPHA_TFIDF = 0.45
BETA_SBERT = 0.45
GAMMA_COMP = 0.10

# Competency extraction settings
TOP_K_COMPETENCIES = 6

# ---------- Utilities ----------
def safe_text(x):
    if pd.isna(x): return ""
    if isinstance(x, (list, tuple)): return " ".join(map(str, x))
    return str(x)

def normalize_whitespace(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def blob_for_course(row: pd.Series) -> str:
    parts = [
        safe_text(row.get("Course_title","")),
        safe_text(row.get("Course_description","")),
        safe_text(row.get("Course_competencies","")),
        safe_text(row.get("Course_department","")),
    ]
    blob = " . ".join([normalize_whitespace(p) for p in parts if p])
    return blob

def parse_credits(x):
    try:
        return float(x)
    except:
        m = re.findall(r"[\d.]+", str(x))
        return float(m[-1]) if m else 0.0

# ---------- Department skill bank (expand as needed) ----------
DEPARTMENT_SKILL_BANK = {
    "CSE": [
        "Software Engineering", "Algorithms", "Data Structures", "Systems Design",
        "Database Design", "Operating Systems", "Distributed Systems",
        "Machine Learning", "Deep Learning", "Model Evaluation",
        "Computer Networks", "Compilers", "Programming Languages"
    ],
    "AM": [
        "Continuum Mechanics", "Finite Element Analysis", "Computational Fluid Dynamics",
        "Structural Dynamics", "Stability Analysis", "Vibration Analysis",
        "Material Modeling", "Numerical Simulation"
    ],
    "CHE": [
        "Transport Phenomena", "Reaction Engineering", "Thermodynamics",
        "Process Design", "Mass & Energy Balances", "Phase Equilibrium",
        "Kinetics", "Process Simulation"
    ],
    "PHY": [
        "Quantum Mechanics", "Electrodynamics", "Statistical Mechanics",
        "Optics", "Solid State Physics", "Experimental Methods"
    ],
    "MATH": [
        "Proof Techniques", "Numerical Methods", "Optimization", "Probability",
        "Linear Algebra", "Differential Equations"
    ],
    "DES": [
        "Design Thinking", "User Experience", "Prototyping", "Interaction Design",
        "Visual Communication", "Human-Centered Design"
    ],
    "HSS": [
        "Critical Thinking", "Ethical Reasoning", "Argumentation", "Qualitative Research",
        "Communication Skills"
    ],
    "GENERAL": [
        "Analytical Reasoning", "Problem Solving", "Communication"
    ],
    # Add more departments as needed
}

# Flatten full skill list for fallback
ALL_SKILLS = list({s for skills in DEPARTMENT_SKILL_BANK.values() for s in skills})

# ---------- Load and preprocess courses ----------
def load_courses(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # ensure columns
    for c in ["Course_id","Course_title","Course_description","Course_competencies",
              "Course_department","Course_slot","Course_credits","Course_prerequisites","Course_instructor"]:
        if c not in df.columns:
            df[c] = ""
    df["__text_blob__"] = df.apply(blob_for_course, axis=1)
    df["Course_competencies"] = df["Course_competencies"].fillna("").astype(str)
    df["Course_credits_num"] = df["Course_credits"].apply(parse_credits)
    # ensure department normalized (if empty)
    df["Course_department"] = df["Course_department"].fillna("GENERAL").astype(str)
    return df

# ---------- SBERT embedding helpers ----------
def load_sbert_model(model_name: str = SBERT_MODEL_NAME):
    print(f"[info] Loading SBERT model: {model_name} ...")
    model = SentenceTransformer(model_name)
    return model

def compute_course_embeddings_sbert(df: pd.DataFrame, model: SentenceTransformer,
                                    cache_path: Optional[str] = EMBEDDINGS_CACHE) -> np.ndarray:
    """
    Compute SBERT embeddings for course text blobs and optionally cache them.
    Returns a numpy array of shape (n_courses, dim)
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            # Check if cached size matches
            if isinstance(cache, dict) and cache.get("n") == len(df):
                print("[info] Loaded SBERT embeddings from cache.")
                return cache["embeddings"]
        except Exception as e:
            print("[warn] Could not load embeddings cache:", e)

    texts = df["__text_blob__"].fillna("").astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    if cache_path:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"n": len(df), "embeddings": embeddings}, f)
            print(f"[info] Cached embeddings to {cache_path}")
        except Exception:
            pass
    return embeddings

# ---------- Competency generator using semantic similarity ----------
def generate_course_competencies_semantic(title: str, description: str, department: str,
                                          model: SentenceTransformer, skill_bank: Dict[str, List[str]],
                                          top_k: int = TOP_K_COMPETENCIES) -> List[str]:
    """
    Use SBERT to map course (title+description) to department skill bank and return top_k competencies.
    """
    dept = (department or "GENERAL").upper()
    # choose skill pool: department-specific then fallback to global
    pool = skill_bank.get(dept, []) + ALL_SKILLS
    pool = list(dict.fromkeys(pool))  # deduplicate preserving order

    # encode course text and pool
    course_text = normalize_whitespace(" ".join([title, description]))
    course_emb = model.encode(course_text, convert_to_numpy=True)
    pool_embs = model.encode(pool, convert_to_numpy=True)
    # compute cosine similarity
    sims = util.cos_sim(course_emb, pool_embs).numpy().flatten()
    # pick top_k skills
    idxs = np.argsort(-sims)[:top_k]
    selected = [pool[i] for i in idxs]
    return selected

# ---------- TF-IDF + SBERT Hybrid Recommender ----------
class SemanticCourseRecommender:
    def __init__(self, course_df: pd.DataFrame,
                 tfidf_vectorizer: Optional[TfidfVectorizer] = None,
                 sbert_model_name: str = SBERT_MODEL_NAME,
                 use_cache_embeddings: bool = True):
        self.df = course_df.reset_index(drop=True)
        self.tfidf_vectorizer = tfidf_vectorizer or TfidfVectorizer(
            max_features=20000, ngram_range=(1,2), stop_words="english")
        self.course_texts = self.df["__text_blob__"].fillna("").astype(str).tolist()
        print("[info] Fitting TF-IDF vectorizer on courses...")
        self.tfidf = self.tfidf_vectorizer.fit_transform(self.course_texts)
        # SBERT model and embeddings
        self.sbert_model = load_sbert_model(sbert_model_name)
        if use_cache_embeddings:
            self.sbert_embeddings = compute_course_embeddings_sbert(self.df, self.sbert_model, EMBEDDINGS_CACHE)
        else:
            self.sbert_embeddings = self.sbert_model.encode(self.course_texts, convert_to_numpy=True)
        # Precompute course competency tags semantically if not already present
        # If Course_competencies is empty or generic, replace with semantic extraction
        self.df["__semantic_competencies__"] = self.df.apply(
            lambda r: generate_course_competencies_semantic(
                r["Course_title"], r["Course_description"], r["Course_department"],
                self.sbert_model, DEPARTMENT_SKILL_BANK, top_k=TOP_K_COMPETENCIES
            ) if (not r["Course_competencies"] or r["Course_competencies"].strip() in ("[]","")) else
            # if existing competencies string, try to parse as list or keep as string tokens
            (json.loads(r["Course_competencies"]) if re.match(r"^\s*\[", str(r["Course_competencies"])) else
             [tok.strip() for tok in re.split(r"[;,|\n]", str(r["Course_competencies"])) if tok.strip()]),
            axis=1
        )

    def build_user_vectors(self, user_profile: Dict[str,Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (tfidf_vector, sbert_vector) for user profile text.
        """
        # Build TF-IDF text
        pieces = []
        pieces.append(" ".join(user_profile.get("User_course_interest", []) if isinstance(user_profile.get("User_course_interest", []), list)
                               else [user_profile.get("User_course_interest","")]))
        pieces.append(" ".join(user_profile.get("User_competencies", []) if isinstance(user_profile.get("User_competencies", []), list)
                               else [user_profile.get("User_competencies","")]))
        pieces.append(user_profile.get("User_career_interest",""))
        prev = user_profile.get("User_previous_courses", [])
        if isinstance(prev, list):
            pieces.append(" ".join(prev))
        else:
            pieces.append(str(prev))
        query = normalize_whitespace(" . ".join([p for p in pieces if p]))
        tfidf_vec = self.tfidf_vectorizer.transform([query])
        sbert_vec = self.sbert_model.encode([query], convert_to_numpy=True)[0]
        return tfidf_vec, sbert_vec

    def competency_overlap_score(self, user_profile: Dict[str,Any], course_idx: int) -> float:
        """
        Compute overlap between user's competencies and course semantic competencies.
        We compute semantic similarity between user's competencies and course competencies
        and return a normalized score in [0,1].
        """
        user_comps = user_profile.get("User_competencies") or []
        if isinstance(user_comps, str):
            user_comps = [user_comps]
        user_comps = [str(x) for x in user_comps if x]

        if not user_comps:
            return 0.0

        # encode both sets
        pool = list(self.df.at[course_idx, "__semantic_competencies__"])
        if not pool:
            return 0.0

        try:
            user_embs = self.sbert_model.encode(user_comps, convert_to_numpy=True)
            pool_embs = self.sbert_model.encode(pool, convert_to_numpy=True)
        except Exception:
            # fallback: simple overlap count
            overlap = len(set(user_comps).intersection(set(pool)))
            return overlap / max(1, len(pool))

        sims = util.cos_sim(user_embs, pool_embs).numpy()
        # For each user competency, take max match in pool; average across user comps
        best_per_user = sims.max(axis=1)  # shape: (len(user_comps),)
        score = float(best_per_user.mean())  # in [0,1]
        return score

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
        Hybrid recommendation combining TF-IDF, SBERT similarity, and competency overlap.
        """
        tfidf_vec, sbert_vec = self.build_user_vectors(user_profile)

        # TF-IDF similarity
        sims_tfidf = cosine_similarity(tfidf_vec, self.tfidf).flatten()
        # SBERT similarity: cosine between sbert_vec and precomputed embeddings
        sbert_embs = self.sbert_embeddings  # shape (n_courses, dim)
        # normalize for numerical stability
        sbert_sim = util.cos_sim(sbert_vec, sbert_embs).numpy().flatten()
        # competency overlap
        comp_scores = np.array([self.competency_overlap_score(user_profile, i) for i in range(len(self.df))])

        # combine
        hybrid_scores = (ALPHA_TFIDF * sims_tfidf) + (BETA_SBERT * sbert_sim) + (GAMMA_COMP * comp_scores)

        # Filtering: slots
        scores = hybrid_scores.copy()
        if exclude_slots:
            mask_slots = self.df["Course_slot"].isin(exclude_slots)
            scores[mask_slots.values] = -1.0

        # department whitelist (if provided)
        if department_whitelist:
            mask_not = ~self.df["Course_department"].isin(department_whitelist)
            scores[mask_not.values] *= 0.5

        # credit bounds
        if min_credits is not None:
            mask = self.df["Course_credits_num"] < min_credits
            scores[mask.values] = -1.0
        if max_credits is not None:
            mask = self.df["Course_credits_num"] > max_credits
            scores[mask.values] = -1.0

        # prerequisites penalty (same logic as before)
        user_prev = {str(x).strip().upper() for x in (user_profile.get("User_previous_courses") or [])}
        for i, prereq in enumerate(self.df["Course_prerequisites"].astype(str).fillna("")):
            if not prereq or prereq.strip() in ("[]", "nan"):
                continue
            tokens = re.findall(r"[A-Z]{2,5}[\s\-]?\d{2,5}", prereq.upper())
            tokens = {t.replace(" ", "").replace("-", "") for t in tokens}
            if tokens and not tokens.intersection(user_prev):
                scores[i] = scores[i] * (1.0 - prereq_penalty)

        # assemble output
        out = self.df.copy()
        out["score"] = scores
        out["tfidf_sim"] = sims_tfidf
        out["sbert_sim"] = sbert_sim
        out["comp_score"] = comp_scores
        out["reason_blob"] = out["__text_blob__"].str.slice(0,300)

        out = out.sort_values("score", ascending=False)
        out = out[out["score"] > 0].head(top_n)
        cols = ["Course_id","Course_title","Course_department","Course_credits_num",
                "Course_slot","Course_instructor","score","tfidf_sim","sbert_sim","comp_score","reason_blob",
                "__semantic_competencies__"]
        return out[cols]

# ---------- Demo ----------
def demo():
    df = load_courses(COURSE_CSV_PATH)
    rec = SemanticCourseRecommender(df)

    user_profile = {
        "User_id": "U1001",
        "User_program": "Btech",
        "User_department": "CSE",
        "User_branch": "COL",
        "User_name": "Anita",
        "User_age": 20,
        "user_year_of_study": 2,
        "User_previous_courses": ["COL1000", "COL106", "COL216"],
        "User_course_interest": ["Databases", "Distributed Systems", "Machine Learning"],
        "User_career_interest": "Data Science",
        "User_competencies": ["Python", "SQL", "Linear Algebra"],
        "User_grades": {"COL1000": "A", "COL106": "A-", "COL216": "B+"},
        "User_occupiedslots": ["A","B"],
        "User_credits_requirement": 12.0,
    }

    recs = rec.recommend(
        user_profile=user_profile,
        top_n=12,
        exclude_slots=user_profile.get("User_occupiedslots", []),
        prereq_penalty=0.25
    )
    pd.set_option("display.max_colwidth", 120)
    print("Top recommendations for", user_profile["User_name"])
    print(recs.to_string(index=False))

if __name__ == "__main__":
    demo()
