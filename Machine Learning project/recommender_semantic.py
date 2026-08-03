# recommender_semantic.py
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

from recommender import generate_course_competencies_semantic

# ---------- CONFIG ----------
COURSE_CSV_PATH = "Courses_dataset_new.xlsx"   # <--- Course dataset path
SBERT_MODEL_NAME = "all-MiniLM-L6-v2" 
EMBEDDINGS_CACHE = "course_embeddings_sbert1.pkl"

ALPHA_TFIDF = 0.45          # weight for TF-IDF similarity
BETA_SBERT = 0.45           # weight for SBERT similarity
GAMMA_COMP = 0.10           # weight for competency similarity  
TOP_K_COMPETENCIES = 6      # number of top course competencies to extract

DEBUG = True  # <--- DEBUG MODE ENABLED

# ---------- UTILS ----------
def debug(*args):
    if DEBUG:
        print(*args)

def safe_text(x):
    # safely convert to string
    if pd.isna(x): return ""
    if isinstance(x, (list, tuple)): return " ".join(map(str, x))
    return str(x)

def normalize_whitespace(s: str) -> str:
    # collapse multiple spaces/newlines
    return re.sub(r"\s+", " ", s).strip()

def blob_for_course(row: pd.Series) -> str:
    # Combine title, description, competencies (if stringified list) and department
    parts = [
        safe_text(row.get("Course_title","")),
        safe_text(row.get("Course_description","")),
        safe_text(row.get("Course_competencies","")),
        safe_text(row.get("Course_department","")),
    ]
    blob = " . ".join([normalize_whitespace(p) for p in parts if p])
    return blob if blob.strip() else "empty"

def parse_credits(x):
    # parse course credits into float
    try:
        return float(x)
    except:
        m = re.findall(r"[\d.]+", str(x))
        return float(m[-1]) if m else 0.0


# ---------- DEPARTMENT SKILL BANK ----------
DEPARTMENT_SKILL_BANK =  {

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

ALL_SKILLS = list({s for skills in DEPARTMENT_SKILL_BANK.values() for s in skills})     # All unique skills, is independent of dept

# ---------- LOAD COURSES WITH DEBUG ----------
def load_courses(csv_path: str) -> pd.DataFrame:
    
    # Auto-detect file format
    if csv_path.lower().endswith(".xlsx"):
        df = pd.read_excel(csv_path, engine="openpyxl")
    else:
        df = pd.read_csv(csv_path)

    debug("\n[DEBUG] Loaded columns:", df.columns.tolist())
    debug(f"[DEBUG] Loaded {len(df)} rows")

    # -------- TITLE / DESCRIPTION FIX --------
    POSSIBLE_TITLE = ["Course_title","course_title","Title","Course Title","title"]
    POSSIBLE_DESC  = ["Course_description","course_description","Description","Course Description"]

    # Fix title
    if "Course_title" not in df.columns:
        for c in POSSIBLE_TITLE:
            if c in df.columns:
                df["Course_title"] = df[c]
                break
        else:
            df["Course_title"] = ""
            debug("[WARN] No title column found. All Course_title set to empty.")

    # Fix description
    if "Course_description" not in df.columns:
        for c in POSSIBLE_DESC:
            if c in df.columns:
                df["Course_description"] = df[c]
                break
        else:
            df["Course_description"] = ""
            debug("[WARN] No description column found. All Course_description empty.")

    # Ensure baseline columns exist
    baseline_cols = [
        "Course_id","Course_competencies","Course_department",
        "Course_slot","Course_credits","Course_prerequisites","Course_instructor"
    ]
    for c in baseline_cols:
        if c not in df.columns:
            df[c] = ""

    # -------- SLOT NAME AUTO-DETECT --------
    SLOT_CANDIDATES = [
        "Slot_Name","slot_name","Slot","slot","SlotName","slotname",
        "SLOT","Slot Name"
    ]

    slot_col = None
    for c in SLOT_CANDIDATES:
        if c in df.columns:
            slot_col = c
            break

    # fallback
    if not slot_col:
        for c in df.columns:
            if "slot" in c.lower():
                slot_col = c
                break

    if slot_col:
        debug(f"[INFO] Using slot column: {slot_col}")
        df["Course_slot"] = df[slot_col]
    else:
        debug("[WARN] No slot column found. Setting Course_slot empty.")
        df["Course_slot"] = ""

    # -------- SLOT FILTERING --------
    before_slot = len(df)
    df = df[df["Course_slot"].astype(str).str.strip() != ""]
    after_slot = len(df)

    debug(f"[DEBUG] Slot filter: kept {after_slot}/{before_slot} rows")

    if after_slot == 0:
        raise ValueError(
            "❌ CRITICAL: No courses have slot info.\n"
            "Check column names. Provide a screenshot of your Excel header."
        )

    # -------- TEXT BLOB DEBUG --------
    df["__text_blob__"] = df.apply(blob_for_course, axis=1)

    debug("\n[DEBUG] Example __text_blob__ values:")
    debug(df["__text_blob__"].head().tolist())

    # Check for empty text blobs
    if all(df["__text_blob__"].str.strip() == "empty"):
        raise ValueError("❌ All text blobs are empty. Fix title/description column names.")
    
    # ---------- COURSE COMPETENCY GENERATION ----------
    model = SentenceTransformer(SBERT_MODEL_NAME)

    df["Course_competencies"] = df.apply(
    lambda row: generate_course_competencies_semantic(
        row["Course_title"],
        row["Course_description"],
        row["Course_department"],
        model,
        DEPARTMENT_SKILL_BANK,
        TOP_K_COMPETENCIES
    ),
    axis=1
    )

    # Recompute blob now that competencies exist
    df["__text_blob__"] = df.apply(blob_for_course, axis=1)


    return df.reset_index(drop=True)


# ---------- SBERT ----------
def load_sbert_model(name=SBERT_MODEL_NAME):
    debug(f"[info] Loading SBERT model: {name}")
    return SentenceTransformer(name)

def compute_course_embeddings_sbert(df, model, cache_path=EMBEDDINGS_CACHE):
    texts = df["__text_blob__"].tolist()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


# ---------- SEMANTIC COMPETENCY EXTRACTOR ----------
def generate_course_competencies_semantic(title, desc, dept, model, bank, top_k=6):
    dept = (dept or "GENERAL").upper()              # Default to GENERAL if missing
    pool = bank.get(dept, []) + ALL_SKILLS          # Combine dept skills + all skills
    pool = list(dict.fromkeys(pool))                # Remove duplicates

    course_text = normalize_whitespace(f"{title} {desc}")         # Combine title + desc
    c_emb = model.encode(course_text, convert_to_numpy=True)      # Course embedding
    p_emb = model.encode(pool, convert_to_numpy=True)             # Pool embeddings

    sims = util.cos_sim(c_emb, p_emb).numpy().flatten()
    idxs = np.argsort(-sims)[:top_k]
    return [pool[i] for i in idxs]


# ---------- RECOMMENDER ----------
class SemanticCourseRecommender:
    def __init__(self, df):
        self.df = df

        debug("[info] Fitting TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1,2),
            stop_words="english"
        )
        self.tfidf = self.vectorizer.fit_transform(self.df["__text_blob__"])

        self.sbert = load_sbert_model()
        self.sbert_embeddings = compute_course_embeddings_sbert(self.df, self.sbert)

    def build_user_vectors(self, user_profile):
        parts = []
        parts.extend(user_profile.get("User_course_interest", []))
        parts.extend(user_profile.get("User_competencies", []))
        parts.append(user_profile.get("User_career_interest",""))
        parts.append(" ".join(user_profile.get("User_previous_courses",[])))

        query = normalize_whitespace(" ".join(parts))
        tfidf_vec = self.vectorizer.transform([query])
        sbert_vec = self.sbert.encode([query], convert_to_numpy=True)[0]
        return tfidf_vec, sbert_vec

    def recommend(self, user_profile, top_n=10, exclude_slots=None):

        tfidf_vec, sbert_vec = self.build_user_vectors(user_profile)

        sims_tfidf = cosine_similarity(tfidf_vec, self.tfidf).flatten()
        sims_sbert = util.cos_sim(sbert_vec, self.sbert_embeddings).numpy().flatten()

        scores = 0.45*sims_tfidf + 0.45*sims_sbert

        # Slot filtering
        if exclude_slots:
            mask = self.df["Course_slot"].isin(exclude_slots)
            scores[mask] = -1.0

        out = self.df.copy()
        out["score"] = scores
        out = out.sort_values("score", ascending=False)
        out = out[out["score"] > 0]

        return out.head(top_n)[["Course_id","Course_title","Course_department","Course_slot","score"]]


# ---------- DEMO ----------
def demo():
    df = load_courses(COURSE_CSV_PATH)
    rec = SemanticCourseRecommender(df)

    user_profile = {
        "User_course_interest": ["Computer Architecture"],
        # "User_competencies": ["Python", "c++","OS concepts"],
        "User_career_interest": "Hardware",
        "User_previous_courses": ["COL216"],
        "User_occupiedslots": ["A"],
    }

    recs = rec.recommend(user_profile, top_n=10, exclude_slots=user_profile["User_occupiedslots"])
    print("\n=== RECOMMENDATIONS ===")
    print(recs.to_string(index=False))


if __name__ == "__main__":
    demo()
