
# Requirements:
# pip install pandas scikit-learn numpy sentence-transformers openpyxl

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
COURSE_CSV_PATH = "courses_populated.xlsx"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SKILL_EMB_CACHE = "skill_embeddings.pkl"
COURSE_EMB_CACHE = "course_text_embeddings.pkl"
COURSE_SBERT_CACHE = "course_embeddings_sbert.pkl"

ALPHA_TFIDF = 0.45
BETA_SBERT = 0.45
TOP_K_COMPETENCIES = 6
GAMMA_POP = 0.10   # weight of popularity score in final ranking


DEBUG = True


# ---------- UTILS ----------
def debug(*args):
    # debug print
    if DEBUG:
        print(*args)


def safe_text(x):
    # Handle lists/arrays first
    if isinstance(x, (list, tuple, np.ndarray)):
        return " ".join(map(str, x))

    # Handle None
    if x is None:
        return ""

    # Handle scalar NaN
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    return str(x)



def normalize_whitespace(s: str) -> str:
    # collapse multiple spaces/newlines
    return re.sub(r"\s+", " ", s).strip()


def blob_for_course(row: pd.Series) -> str:
    # Combine title, description, competencies (if stringified list) and department
    parts = [
        safe_text(row.get("Course_title", "")),
        safe_text(row.get("Course_description", "")),
        safe_text(row.get("Course_competencies", "")),
        safe_text(row.get("Course_department", "")),
    ]
    parts = [normalize_whitespace(p) for p in parts if p.strip()]
    return " . ".join(parts) if parts else "empty"


# ---------- YOUR DEPARTMENT SKILL BANK ----------
# (YOUR FULL SKILL BANK PASTED EXACTLY AS YOU PROVIDED)

DEPARTMENT_SKILL_BANK = {

    "Computer Science and Engineering": [
        "software", "algorithms", "datastructures", "operatingsystems", "databases",
        "distributed", "networks", "cloud", "devops", "compilers",
        "cybersecurity", "parallelcomputing", "concurrency", "programming",
        "versioncontrol", "virtualization", "containers", "webdevelopment",
        "mobiledevelopment", "microservices", "scalability", "optimization",
        "debugging", "testing", "qualityassurance", "ci", "cd",
        "machinelearning", "deeplearning", "nlp", "computervision",
        "recommendations", "informationretrieval", "hpc", "datamining",
        "softwarearchitecture", "designpatterns", "systemdesign",
        "memorymanagement", "networkprotocols", "apidesign",
        "distributedstorage", "virtualmachines", "scheduling",
        "loadbalancing", "faulttolerance", "reliability",
        "performance", "coding"
    ],

    "Applied Mechanics": [
        "continuum", "finiteelement", "fracture", "stressanalysis", "strain",
        "mechanics", "elasticity", "plasticity", "dynamics", "vibrations",
        "harmonic", "modal", "nonlinear", "fsi", "materialmodeling",
        "computational", "simulation", "meshing", "topologyoptimization",
        "structuralanalysis", "fatigue", "creep", "buckling",
        "stability", "impactanalysis", "contactmechanics", "tensors",
        "kinematics", "statics", "thermomechanics", "fluidmechanics",
        "rheology", "numericalmethods", "boundarymethods",
        "energyprinciples", "variational", "mathematicalmodeling",
        "solverdesign", "discretization", "approximation",
        "multiphysics", "damping", "oscillation", "loading",
        "crackpropagation", "damage", "composites",
        "optimization", "validation", "calibration"
    ],

    "GENERAL": [
        "analytical", "reasoning", "criticalthinking", "communication",
        "teamwork", "leadership", "presentation", "adaptability", "creativity",
        "workethic", "professionalism", "problemsolving", "logic",
        "collaboration", "time_management", "planning",
        "organization", "initiative", "writing", "documentation",
        "research", "analysis", "attention", "focus",
        "decisionmaking", "empathy", "persuasion", "negotiation",
        "resilience", "learning", "curiosity", "selfmanagement",
        "innovation", "brainstorming", "reviewing", "feedback",
        "structuring", "prioritization", "conceptualization",
        "coordination", "socialskills", "reflection",
        "ethics", "confidence", "multitasking",
        "listening", "motivation", "synthesis",
        "comprehension", "collaboration"
    ],

    "physics": [
        "mechanics", "electromagnetism", "quantum", "statistical", "solidstate",
        "optics", "photonics", "thermodynamics", "relativity", "waves",
        "oscillations", "atomicphysics", "nuclearphysics", "particlephysics",
        "astrophysics", "cosmology", "plasmaphysics", "condensedmatter",
        "superconductivity", "semiconductors", "lasers", "spectroscopy",
        "microscopy", "fluiddynamics", "nonlinearphysics", "computational",
        "simulation", "modeling", "mathematicalphysics", "measurements",
        "instrumentation", "quantumoptics", "optoelectronics",
        "nanophotonics", "nanoscience", "quantumcomputing",
        "opticalcommunication", "fiberoptics", "interferometry",
        "holography", "radiation", "scattering", "thermoacoustics",
        "optomechanics", "crystallography", "diffraction",
        "statistics", "acoustics", "luminescence",
        "magnetism", "spintronics"
    ],

    "Mathematics": [
        "calculus", "algebra", "differentialequations", "probability",
        "statistics", "realanalysis", "complexanalysis", "optimization",
        "numericalmethods", "topology", "geometry", "graph", "logic",
        "settheory", "categorytheory", "stochastic", "markov",
        "combinatorics", "numbertheory", "discrete", "gameTheory",
        "operationsresearch", "measuretheory", "functional",
        "tensoranalysis", "variational", "matrixalgebra",
        "linearsystems", "vectorcalculus", "dynamicalsystems",
        "fourier", "laplace", "pde", "ode", "modelling",
        "finiteelement", "approximation", "interpolation",
        "regression", "inference", "randomvariables",
        "probabilitytheory", "cryptomath", "symbolic",
        "chaos", "fractals", "continuum",
        "transformations", "dimensionreduction", "algorithms",
        "booleanlogic", "linearprogramming"
    ],

    "SIT": [
        "cryptography", "pentesting", "vulnerabilities", "threatmodeling",
        "riskanalysis", "incidentresponse", "malware", "forensics",
        "reversing", "identity", "accesscontrol", "networksecurity",
        "firewalls", "ids", "ips", "honeypots", "phishing",
        "osint", "authentication", "authorization", "securecoding",
        "zerotrust", "siem", "monitoring", "compliance",
        "governance", "encryption", "hashing", "keymanagement",
        "protocolsecurity", "websecurity", "cloudsecurity",
        "containersecurity", "endpointsecurity", "datasecurity",
        "redteaming", "blueteaming", "socialengineering",
        "threatintel", "privilegeescalation", "mitre",
        "ddos", "botnets", "rootkits", "ransomware",
        "cryptanalysis", "networkforensics", "loganalysis",
        "binaryanalysis", "sandboxing"
    ],

    "School of AI": [
        "machinelearning", "deeplearning", "reinforcement", "nlp", "vision",
        "speech", "optimization", "explainableai", "generative",
        "transformers", "cnn", "rnn", "lstm", "gans",
        "probabilistic", "bayesian", "clustering", "classification",
        "regression", "dimensionalityreduction", "featureengineering",
        "reasoning", "knowledgegraphs", "search", "planning",
        "robotics", "autonomy", "simulation", "datascience",
        "analytics", "interpretability", "datascaling",
        "multimodal", "federatedlearning", "ondeviceai",
        "representations", "embeddings", "prompting",
        "evaluation", "inference", "deployment",
        "computervision", "objectdetection", "segmentation",
        "textgeneration", "agents", "optimizationalgorithms",
        "datapreprocessing", "hyperparameter", "neuralarchitecture"
    ],

    "Civil Engineering": [
        "structural", "geotechnical", "surveying", "hydrology", "transportation",
        "environmental", "construction", "concrete", "steel", "timber",
        "geology", "foundation", "earthquake", "waterresources",
        "hydraulics", "infrastructure", "planning", "urban", "architecture",
        "estimation", "projectmanagement", "contracting", "roads",
        "bridges", "tunnels", "dams", "canals", "sewage",
        "wastewater", "geospatial", "mapping", "designcodes",
        "soilmechanics", "laboratorytesting", "simulation",
        "sustainability", "materials", "transportplanning",
        "pavementdesign", "surveyinstruments", "geodesy",
        "remotesensing", "stability", "loadanalysis",
        "drainage", "irrigation", "buildingphysics",
        "groundimprovement", "riskanalysis"
    ],

    "Mechanical Engineering": [
        "thermodynamics", "heattransfer", "fluids", "cad", "cam",
        "manufacturing", "machinedesign", "robotics", "automotive",
        "aerodynamics", "kinematics", "dynamics", "mechatronics",
        "materialscience", "energysystems", "hvac", "refrigeration",
        "icengines", "turbomachinery", "vibrations", "controls",
        "tribology", "computational", "fem", "fatigue",
        "casting", "welding", "machining", "additivemanufacturing",
        "qualitycontrol", "metrology", "instrumentation",
        "thermoacoustics", "nanomaterials", "composites",
        "designoptimization", "structures", "simulation",
        "lubrication", "thermofluids", "pumps", "compressors",
        "hydraulics", "pneumatics", "ergonomics",
        "packaging", "mechanisms", "failureanalysis"
    ],

    "Centre for Biomedical Engineering": [
        "molecular", "genetics", "bioprocess", "microbiology",
        "bioinformatics", "cellculture", "fermentation", "immunology",
        "genomics", "proteomics", "transcriptomics", "sequencing",
        "biomaterials", "bioreactors", "analyticalbiology",
        "enzymology", "metabolomics", "crystallography",
        "drugdevelopment", "vaccines", "cellengineering",
        "tissueengineering", "syntheticbiology", "nanobiotech",
        "biostatistics", "biochemistry", "biomechanics",
        "toxicology", "pathology", "microbialgenetics",
        "bioremediation", "biomarkers", "assaydevelopment",
        "cloning", "crispr", "signaltransduction",
        "virology", "bacteriology", "parasitology",
        "biophysics", "chemoinformatics", "biosensors",
        "agrobiotech", "proteinengineering", "industrialbiotech",
        "bioproduction", "downstreamprocessing"
    ],

    "DESIGN": [
        "designthinking", "ux", "ui", "hcd", "visual", "creativity",
        "graphics", "storytelling", "typography", "ergonomics",
        "prototyping", "cad", "illustration", "animation",
        "interactiondesign", "productdesign", "industrialdesign",
        "sketching", "composition", "color", "branding",
        "aesthetics", "formdevelopment", "usability",
        "research", "wireframing", "mockups", "modeling",
        "packaging", "fabrication", "creativecoding",
        "motiongraphics", "trendanalysis", "innovation",
        "materials", "simulation", "cognitive",
        "responsivedesign", "servicedesign", "specification",
        "visualization", "rendering", "interfacearchitecture",
        "materialexploration", "userflows", "layout",
        "3dmodeling", "exhibitiondesign"
    ],

    "Textile and Fibre Engineering": [
        "fiberscience", "polymers", "yarn", "weaving", "knitting",
        "dyeing", "printing", "finishing", "testing", "quality",
        "apparel", "fashiontech", "nonwoven", "spinning",
        "blending", "carding", "combing", "twisting",
        "weavemechanics", "fabricdesign", "textilechemistry",
        "functionaltextiles", "nanotextiles", "smarttextiles",
        "coating", "lamination", "characterization", "microscopy",
        "merchandising", "cadtextile", "fiberanalysis",
        "performance", "sustainability", "biotextiles",
        "technicaltextiles", "tensiletesting", "colorfastness",
        "elasticity", "finishingchemistry", "garmentconstruction",
        "denim", "textilephysics", "patternmaking",
        "textilemachinery", "qualitycontrol", "drapability",
        "absorbency", "spinningtechnology"
    ],

    "Humanities and Social Sciences": [
        "criticalthinking", "ethics", "sociology", "psychology",
        "philosophy", "communication", "writing", "history",
        "culture", "anthropology", "politics", "economics",
        "behavior", "literature", "linguistics",
        "reasoning", "creativity", "leadership",
        "education", "aesthetics", "analysis",
        "logic", "consciousness", "research",
        "community", "identity", "values",
        "morality", "negotiation", "persuasion",
        "empathy", "relation", "globalization",
        "ethnography", "media", "journalism",
        "publicspeaking", "conflictresolution",
        "philosophicalanalysis", "civics",
        "rhetoric", "writinganalysis",
        "interpretation", "ethicsreview",
        "humanbehavior", "reasoninglogic",
        "discourse", "inquiry", "socialsystems"
    ],

    "Optics and Photonics Centre": [
        "optics", "photonics", "lasers", "quantumoptics", "spectroscopy",
        "microscopy", "fiberoptics", "nanophotonics", "optoelectronics",
        "nonlinearoptics", "interferometry", "holography",
        "lithography", "opticalcommunication", "waveguides",
        "opticaldesign", "thinfilms", "diffraction", "scattering",
        "polarization", "imaging", "sensoranalysis", "metamaterials",
        "plasmonics", "lensdesign", "fourieroptics",
        "radiometry", "photometry", "opticaltesting",
        "laserprocessing", "optomechanics", "adaptiveoptics",
        "quantumdots", "siliconphotonics", "detectors",
        "femtosecond", "ultrafast", "opticalcoherence",
        "spectralanalysis", "phaseshifting", "opticalmodeling",
        "opticalfibers", "waveoptics", "beampropagation",
        "opticalcomponents", "opticalfabrication",
        "opticalfilters", "opticalmodulators",
        "opticalamplifiers", "opticalmetrology"
    ],

    "National Resource Centre for Value Education in Engineering": [
        "ethics", "values", "moralreasoning", "responsibility",
        "sustainability", "leadership", "empathy", "holisticthinking",
        "awareness", "culture", "professionalism", "integrity",
        "honesty", "trust", "discipline", "selfreflection",
        "service", "compassion", "mindfulness", "respect",
        "tolerance", "cooperation", "citizenship", "inclusion",
        "teamspirit", "humaneness", "gratitude", "kindness",
        "positivity", "wellbeing", "accountability", "adaptability",
        "purpose", "commitment", "fairness", "nonviolence",
        "ethicalliteracy", "workethic", "attitude",
        "selfcontrol", "emotionalintelligence", "socialresponsibility",
        "communityengagement", "lifeskills", "decisionmaking",
        "virtues", "transparency", "equity",
        "humility", "openmindedness"
    ],

    "Chemical Engineering": [
    "massbalance", "energybalance", "thermodynamics", "fluidmechanics",
    "heattransfer", "masstransfer", "reactionengineering", "processdesign",
    "processsimulation", "processeconomics", "processcontrol", "processsafety",
    "chemicalkinetics", "catalysis", "reactordesign", "separations",
    "distillation", "absorption", "adsorption", "extraction",
    "crystallization", "evaporation", "filtration", "membranetech",
    "pipedesign", "heatexchangers", "pumppiping", "materialbalance",
    "transportphenomena", "computationalfluiddynamics", "polymers",
    "biochemicalengineering", "environmentalengineering",
    "wastetreatment", "energysystems", "nanomaterials",
    "processoptimization", "scaleup", "industrialchemistry",
    "petrochemical", "refining", "corrosion", "electrochemical",
    "hazardanalysis", "instrumentation", "plantoperations",
    "qualitycontrol", "sustainability", "greenengineering",
    "processmodelling", "engineeringmaterials"
], 

    "Chemistry": [
    "organicchemistry", "inorganicchemistry", "physicalchemistry", "analyticalchemistry",
    "biochemistry", "quantumchemistry", "thermochemistry", "electrochemistry", "spectroscopy",
    "chromatography", "massspectrometry", "nmr", "ftir", "xrd",
    "kinetics", "thermodynamics", "quantummechanics", "reactionmechanisms",
    "stoichiometry", "molecularmodeling", "computationalchemistry", "crystallography",
    "polymers", "supramolecular", "nanochemistry", "photochemistry",
    "catalysis", "surfacechemistry", "colloids", "materialschemistry",
    "environmentalchemistry", "greenchemistry", "toxicology", "chemometrics",
    "solids", "liquids", "gases", "solutionchemistry",
    "acidbase", "redox", "coordinationchemistry", "organometallic",
    "separationtechniques", "labinstrumentation", "synthesis", "purification",
    "biopolymers", "enzymechemistry", "energetics", "chemicalbonding"
], 
    "Electrical Engineering": [
    "circuittheory", "analogelectronics", "digitalelectronics", "signalsystems",
    "signalprocessing", "microprocessors", "microcontrollers", "embedded",
    "electromagnetics", "powersystems", "powergeneration", "transmission",
    "distribution", "smartgrid", "hvdc", "machines",
    "transformers", "motors", "drives", "controlsystems",
    "automation", "instrumentation", "sensors", "actuators",
    "wirelesscommunication", "rfdesign", "microwaveengineering", "antennas",
    "telemetry", "modulation", "demodulation", "vlsi",
    "pcbdesign", "spicemodelling", "analogdesign", "digitaldesign",
    "renewableenergy", "batterymanagement", "powerquality", "filterdesign",
    "protection", "relays", "faultanalysis", "electricvehicles",
    "robotics", "industrialautomation", "scada", "iot",
    "opamps", "electronicsmanufacturing"
], 

    "Materials Science and Engineering": [
    "crystallography", "microstructure", "defects", "dislocations",
    "phaseequilibria", "phasetransformations", "thermodynamics",
    "kinetics", "solidification", "diffusion",
    "metallurgy", "polymers", "ceramics", "composites",
    "nanomaterials", "nanostructures", "semiconductors",
    "electronicmaterials", "biomaterials", "magneticmaterials",
    "optoelectronicmaterials", "energymaterials",
    "failureanalysis", "fracturemechanics", "fatigue",
    "corrosion", "oxidation", "coatings",
    "surfaceengineering", "heattreatment", "powderprocessing",
    "additivemanufacturing", "thinfilms", "deformation",
    "hardness", "toughness", "creep",
    "tribology", "alloydesign", "materialcharacterization",
    "xrd", "sem", "tem", "spectroscopy",
    "differentialscanningcalorimetry", "thermogravimetric",
    "computationalmaterials", "materialsmodelling",
    "processoptimization", "microfabrication"
], 
    
    "Centre for Applied Research in Electronics": [
    "analogelectronics", "digitalelectronics", "circuitdesign", "microelectronics",
    "vlsi", "semiconductordevices", "devicefabrication", "mems",
    "nanodevices", "embedded", "microcontrollers", "fpga",
    "signalprocessing", "dsp", "filters", "sensors",
    "instrumentation", "dataacquisition", "transducers",
    "opamps", "rfdesign", "microwaveengineering", "antennadesign",
    "rfcircuits", "wirelesscommunication", "modulation",
    "communicationcircuits", "lowpowerdesign", "powersupplydesign",
    "mixedsignal", "pcbdesign", "electromagnetics",
    "optoelectronics", "photonics", "laserelectronics",
    "imaging", "opticalsensors", "spectroscopy",
    "reliabilitytesting", "failureanalysis", "electronicpackaging",
    "embeddedcommunication", "iot", "roboticsintegration",
    "machinevision", "signalconditioning", "analoglayout",
    "digitaldesign", "cmosdesign", "highfrequencycircuits",
    "sensorfusion", "hardwareacceleration"
]

}

def compute_popularity_score(rating, enrolled, max_enrolled=10000) -> float:
    """Compute popularity using normalized rating + enrollment."""
    # Handle missing values
    if pd.isna(rating):
        rating = 3.0 
    if pd.isna(enrolled):
        enrolled = 0

    # --- Normalize rating to 0–1 ---
    rating_norm = max(0, min(1, (rating - 1) / 4))  # maps 1→0, 5→1

    # --- Log-scale normalize enrollments ---
    # Prevents huge courses from dominating
    enrolled_norm = np.log1p(enrolled) / np.log1p(max_enrolled)

    # Weighted blend 
    popularity = (0.7 * rating_norm) + (0.3 * enrolled_norm)

    return popularity



# ---------- PRECOMPUTATION: SKILL EMBEDDINGS ----------
def load_skill_embeddings(model):
    if os.path.exists(SKILL_EMB_CACHE):
        debug("[cache] Loading skill embeddings from cache...")
        return pickle.load(open(SKILL_EMB_CACHE, "rb"))

    debug("[build] Computing skill embeddings...")
    all_skills = list({skill for skills in DEPARTMENT_SKILL_BANK.values() for skill in skills})

    emb = model.encode(all_skills, convert_to_numpy=True, show_progress_bar=True)
    skill_emb = {all_skills[i]: emb[i] for i in range(len(all_skills))}

    pickle.dump(skill_emb, open(SKILL_EMB_CACHE, "wb"))
    return skill_emb


# ---------- PRECOMPUTATION: COURSE TEXT EMBEDDINGS ----------
def load_course_text_embeddings(texts, model):
    if os.path.exists(COURSE_EMB_CACHE):
        debug("[cache] Loading course text embeddings...")
        return pickle.load(open(COURSE_EMB_CACHE, "rb"))

    debug("[build] Computing course text embeddings...")
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    pickle.dump(emb, open(COURSE_EMB_CACHE, "wb"))
    return emb


# ---------- FAST COMPETENCY GENERATOR ----------
def semantic_competencies_fast(course_emb, dept, bank, skill_embs, top_k):
    # normalize department
    if dept not in bank:
        dept = "GENERAL"

    skills = bank[dept] + list(skill_embs.keys())
    skills = list(dict.fromkeys(skills))

    emb_matrix = np.vstack([skill_embs[s] for s in skills])
    sims = util.cos_sim(course_emb, emb_matrix).numpy().flatten()

    top_idx = np.argsort(-sims)[:top_k]
    return [skills[i] for i in top_idx]


# ---------- LOAD COURSES ----------
def load_courses(csv_path: str) -> pd.DataFrame:

    # Load file
    if csv_path.lower().endswith(".xlsx"):
        df = pd.read_excel(csv_path, engine="openpyxl")
    else:
        df = pd.read_csv(csv_path)

    debug("\n[DEBUG] Loaded columns:", df.columns.tolist())
    debug(f"[DEBUG] Loaded {len(df)} rows")

    # Fix title + description
    POSSIBLE_TITLE = ["Course_title", "course_title", "Title", "Course Title", "title"]
    POSSIBLE_DESC = ["Course_description", "course_description", "Description", "Course Description"]

    if "Course_title" not in df.columns:
        for c in POSSIBLE_TITLE:
            if c in df.columns: df["Course_title"] = df[c]; break
        else:
            df["Course_title"] = ""

    if "Course_description" not in df.columns:
        for c in POSSIBLE_DESC:
            if c in df.columns: df["Course_description"] = df[c]; break
        else:
            df["Course_description"] = ""

    # ---- POPULARITY SCORE (based on course_rating column) ----
    if "course_rating" not in df.columns:
        debug("[WARN] No course_rating column found. Setting neutral popularity.")
        df["course_rating"] = 3.0

    max_enrolled = df["students_enrolled_tilldate"].max()
    df["popularity_score"] = df.apply(
    lambda row: compute_popularity_score(row["course_rating"], row["students_enrolled_tilldate"], max_enrolled),
    axis=1)#computes popularity score from previously enrolled course ratings

    # Ensure required columns
    baseline = ["Course_id", "Course_competencies", "Course_department",
                "Course_slot", "Course_credits", "Course_prerequisites", "Course_instructor"]
    for col in baseline:
        if col not in df.columns:
            df[col] = ""

    # Auto-detect slot column
    slot_col = None
    for c in df.columns:
        if "slot" in c.lower():
            slot_col = c
            break

    if slot_col:
        df["Course_slot"] = df[slot_col]
    else:
        df["Course_slot"] = ""

    before = len(df)
    df = df[df["Course_slot"].astype(str).str.strip() != ""]
    debug(f"[DEBUG] Slot filter kept {len(df)}/{before}")

    if len(df) == 0:
        raise ValueError("❌ No course slots found.")

    # SBERT Model
    model = SentenceTransformer(SBERT_MODEL_NAME)

    # Load skill embeddings
    skill_embs = load_skill_embeddings(model)

    # Compute course text embeddings (title+desc only)
    df["course_text"] = (df["Course_title"].astype(str).fillna("") + " " + df["Course_description"].astype(str).fillna(""))
    df["course_text"] = df["course_text"].apply(lambda x: normalize_whitespace(safe_text(x)))

    course_embs = load_course_text_embeddings(df["course_text"].tolist(), model)
    df["course_emb"] = list(course_embs)

    debug("[INFO] Generating semantic competencies (FAST)...")

    # Generate competencies using only vector math
    df["Course_competencies"] = df.apply(
        lambda row: semantic_competencies_fast(
            row["course_emb"],
            row["Course_department"],
            DEPARTMENT_SKILL_BANK,
            skill_embs,
            TOP_K_COMPETENCIES
        ),
        axis=1
    )

    # Build final text blob
    df["__text_blob__"] = df.apply(blob_for_course, axis=1)

    return df.reset_index(drop=True)


# ---------- SBERT ENCODING FOR RECOMMENDER ----------
def load_sbert_course_embeddings(df, model):
    if os.path.exists(COURSE_SBERT_CACHE):
        debug("[cache] Loading SBERT embeddings for courses...")
        return pickle.load(open(COURSE_SBERT_CACHE, "rb"))

    debug("[build] SBERT embeddings for course blobs...")
    emb = model.encode(df["__text_blob__"].tolist(),
                       convert_to_numpy=True,
                       show_progress_bar=True)

    pickle.dump(emb, open(COURSE_SBERT_CACHE, "wb"))
    return emb


# ---------- RECOMMENDER ----------
class SemanticCourseRecommender:
    def __init__(self, df):
        self.df = df

        debug("[info] Fitting TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.tfidf = self.vectorizer.fit_transform(self.df["__text_blob__"])

        self.sbert = SentenceTransformer(SBERT_MODEL_NAME)
        self.sbert_embeddings = load_sbert_course_embeddings(self.df, self.sbert)

    def build_user_vectors(self, user_profile):
        parts = []
        parts.extend(user_profile.get("User_course_interest", []))
        parts.extend(user_profile.get("User_competencies", []))
        parts.append(user_profile.get("User_career_interest", ""))
        parts.append(" ".join(user_profile.get("User_previous_courses", [])))

        query = normalize_whitespace(" ".join(parts))

        tfidf_vec = self.vectorizer.transform([query])
        sbert_vec = self.sbert.encode([query], convert_to_numpy=True)[0]

        return tfidf_vec, sbert_vec

    def recommend(self, user_profile, top_n=10, exclude_slots=None):

        tfidf_vec, sbert_vec = self.build_user_vectors(user_profile)

        sims_tfidf = cosine_similarity(tfidf_vec, self.tfidf).flatten()
        sims_sbert = util.cos_sim(sbert_vec, self.sbert_embeddings).numpy().flatten()

        scores = (ALPHA_TFIDF * sims_tfidf +BETA_SBERT * sims_sbert +GAMMA_POP * self.df["popularity_score"].values)


        if exclude_slots:
            mask = self.df["Course_slot"].isin(exclude_slots)
            scores[mask] = -1.0

        out = self.df.copy()
        out["score"] = scores
        out = out.sort_values("score", ascending=False)
        out = out[out["score"] > 0]

        return out.head(top_n)[[
            "Course_id", "Course_title",
            "Course_department", "Course_slot", "score"
        ]]


# ---------- DEMO ----------
def demo():
    df = load_courses(COURSE_CSV_PATH)
    rec = SemanticCourseRecommender(df)

    user_profile = {
        "User_course_interest": ["Cloud Systems","Devops","Databases","Security","Development"],
        "User_career_interest": "Cloud and Devops", #only one value allowed
        "User_competencies": ["programming", "python","docker","kubernetes","aws","ci/cd","databases"],
        "User_previous_courses": ["COL216"],
        "User_occupiedslots": ["A"],
    }

    recs = rec.recommend(
        user_profile,
        top_n=10,
        exclude_slots=user_profile["User_occupiedslots"]
    )
    print("\n=== RECOMMENDATIONS ===")
    print(recs.to_string(index=False))


if __name__ == "__main__":
    demo()
