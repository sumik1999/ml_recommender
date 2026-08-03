# recommender_semantic.py
# Fully Optimized + Persistent Caching Version

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

# ========= CONFIG =========
COURSE_CSV_PATH = "trial/Courses_dataset_new-2.xlsx"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Cache files
SKILL_EMB_CACHE = "skill_embeddings.pkl"
COURSE_TEXT_EMB_CACHE = "course_text_embeddings.pkl"
COURSE_SBERT_EMB_CACHE = "course_embeddings_sbert.pkl"
TFIDF_VECTORIZER_CACHE = "tfidf_vectorizer.pkl"
TFIDF_MATRIX_CACHE = "tfidf_matrix.pkl"
PROCESSED_DF_CACHE = "processed_courses.pkl"

TOP_K_COMPETENCIES = 6
ALPHA_TFIDF = 0.45
BETA_SBERT = 0.45
DEBUG = True


# ========= UTILS =========
def debug(*args):
    if DEBUG: print(*args)

def safe_text(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return " ".join(map(str, x))
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)

def normalize_whitespace(s):
    return re.sub(r"\s+", " ", s).strip()

def blob_for_course(row):
    parts = [
        safe_text(row.get("Course_title", "")),
        safe_text(row.get("Course_description", "")),
        safe_text(row.get("Course_competencies", "")),
        safe_text(row.get("Course_department", "")),
    ]
    return " . ".join([normalize_whitespace(p) for p in parts if p.strip()])


# ========= SKILL BANK (paste your full dictionary here) =========
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
],      "GENERAL": [
        "analytical", "reasoning", "criticalthinking", "communication",
        "teamwork", "leadership", "presentation", "adaptability", "creativity",
        "workethic", "professionalism", "problemsolving", "logic",
        "collaboration", "time_management", "planning",
        "organization", "initiative", "writing", "documentation",
        "research", "analysis", "attention", "focus",
        "decisionmaking", "empathy", "persuasion", "negotiation"]

}


# ========= LOAD / BUILD SKILL EMBEDDINGS =========
def load_skill_embeddings(model):
    if os.path.exists(SKILL_EMB_CACHE):
        debug("[cache] Skill embeddings loaded.")
        return pickle.load(open(SKILL_EMB_CACHE, "rb"))

    debug("[build] Computing skill embeddings...")
    all_skills = list({s for skills in DEPARTMENT_SKILL_BANK.values() for s in skills})
    emb = model.encode(all_skills, convert_to_numpy=True, show_progress_bar=True)
    emb_dict = {all_skills[i]: emb[i] for i in range(len(all_skills))}

    pickle.dump(emb_dict, open(SKILL_EMB_CACHE, "wb"))
    return emb_dict


# ========= FAST COMPETENCY GENERATOR =========
def semantic_competencies_fast(course_emb, dept, bank, skill_embs, top_k):
    if dept not in bank:
        dept = "GENERAL"

    skills = list(dict.fromkeys(bank[dept] + list(skill_embs.keys())))
    emb_matrix = np.vstack([skill_embs[s] for s in skills])
    sims = util.cos_sim(course_emb, emb_matrix).numpy().flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return [skills[i] for i in top_idx]


# ========= LOAD OR CREATE PROCESSED COURSE FRAME =========
def load_processed_courses(model):
    if os.path.exists(PROCESSED_DF_CACHE):
        debug("[cache] Processed DF loaded.")
        return pickle.load(open(PROCESSED_DF_CACHE, "rb"))

    # ------------- RAW LOAD ----------------
    df = pd.read_excel(COURSE_CSV_PATH)

    # Fix columns
    # df["Course_title"] = df.get("Course_title", df.get("Title", "")).astype(str)
    # df["Course_description"] = df.get("Course_description", df.get("Description", "")).astype(str)
    # ---- SAFE TITLE ----
    if "Course_title" in df.columns:
        df["Course_title"] = df["Course_title"].astype(str)
    elif "Title" in df.columns:
        df["Course_title"] = df["Title"].astype(str)
    else:
        debug("[WARN] No Course_title or Title column found. Filling with empty string.")
        df["Course_title"] = ""

    # ---- SAFE DESCRIPTION ----
    if "Course_description" in df.columns:
        df["Course_description"] = df["Course_description"].astype(str)
    elif "Description" in df.columns:
        df["Course_description"] = df["Description"].astype(str)
    else:
        debug("[WARN] No Course_description or Description column found. Filling with empty string.")
        df["Course_description"] = ""
    df["Course_department"] = df.get("Course_department", "").astype(str)
    df["Course_slot"] = df[[c for c in df.columns if "slot" in c.lower()][0]]

    # Remove empty slot rows
    df = df[df["Course_slot"].astype(str).str.strip() != ""]

    # Safe course text
    df["course_text"] = (
        df["Course_title"].astype(str) + " " +
        df["Course_description"].astype(str)
    ).apply(lambda x: normalize_whitespace(safe_text(x)))

    # ---------- COURSE TEXT EMBEDDING CACHE ----------
    if os.path.exists(COURSE_TEXT_EMB_CACHE):
        debug("[cache] Course text embeddings loaded.")
        course_text_emb = pickle.load(open(COURSE_TEXT_EMB_CACHE, "rb"))
    else:
        debug("[build] Encoding course text embeddings...")
        course_text_emb = model.encode(df["course_text"].tolist(),
                                       convert_to_numpy=True,
                                       show_progress_bar=True)
        pickle.dump(course_text_emb, open(COURSE_TEXT_EMB_CACHE, "wb"))

    df["course_emb"] = list(course_text_emb)

    # ---------- SKILL EMBEDDINGS ----------
    skill_embs = load_skill_embeddings(model)

    # ---------- FAST COMPETENCY EXTRACTION ----------
    debug("[INFO] Extracting competencies (fast)...")
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

    # Build final blob
    df["__text_blob__"] = df.apply(blob_for_course, axis=1)

    # Save processed DF
    pickle.dump(df, open(PROCESSED_DF_CACHE, "wb"))
    return df


# ========= TF-IDF CACHE BUILDER =========
def load_tfidf(df):
    if os.path.exists(TFIDF_VECTORIZER_CACHE) and os.path.exists(TFIDF_MATRIX_CACHE):
        debug("[cache] TF-IDF vectorizer & matrix loaded.")
        vectorizer = pickle.load(open(TFIDF_VECTORIZER_CACHE, "rb"))
        matrix = pickle.load(open(TFIDF_MATRIX_CACHE, "rb"))
        return vectorizer, matrix

    debug("[build] Building TF-IDF vectorizer…")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    matrix = vectorizer.fit_transform(df["__text_blob__"])

    pickle.dump(vectorizer, open(TFIDF_VECTORIZER_CACHE, "wb"))
    pickle.dump(matrix, open(TFIDF_MATRIX_CACHE, "wb"))
    return vectorizer, matrix


# ========= SBERT COURSE EMBEDDINGS (BLOB) =========
def load_sbert_course_embeddings(df, model):
    if os.path.exists(COURSE_SBERT_EMB_CACHE):
        debug("[cache] SBERT course blob embeddings loaded.")
        return pickle.load(open(COURSE_SBERT_EMB_CACHE, "rb"))

    debug("[build] Encoding SBERT embeddings for full blobs...")
    emb = model.encode(df["__text_blob__"].tolist(),
                       convert_to_numpy=True,
                       show_progress_bar=True)

    pickle.dump(emb, open(COURSE_SBERT_EMB_CACHE, "wb"))
    return emb


# ========= RECOMMENDER =========
class SemanticCourseRecommender:

    def __init__(self, df):
        self.df = df
        self.model = SentenceTransformer(SBERT_MODEL_NAME)

        # Load precomputed components
        self.vectorizer, self.tfidf = load_tfidf(df)
        self.sbert_embeddings = load_sbert_course_embeddings(df, self.model)

    def build_user_vectors(self, profile):
        parts = []
        parts.extend(profile.get("User_course_interest", []))
        parts.extend(profile.get("User_competencies", []))
        parts.append(profile.get("User_career_interest", ""))
        parts.append(" ".join(profile.get("User_previous_courses", [])))

        query = normalize_whitespace(" ".join(parts))
        tfidf_vec = self.vectorizer.transform([query])
        sbert_vec = self.model.encode([query], convert_to_numpy=True)[0]
        return tfidf_vec, sbert_vec

    def recommend(self, profile, top_n=10, exclude_slots=None):

        tfidf_vec, sbert_vec = self.build_user_vectors(profile)

        sims_tfidf = cosine_similarity(tfidf_vec, self.tfidf).flatten()
        sims_sbert = util.cos_sim(sbert_vec, self.sbert_embeddings).numpy().flatten()

        scores = ALPHA_TFIDF*sims_tfidf + BETA_SBERT*sims_sbert

        if exclude_slots:
            mask = self.df["Course_slot"].isin(exclude_slots)
            scores[mask] = -1e9

        out = self.df.copy()
        out["score"] = scores
        out = out.sort_values("score", ascending=False)
        out = out[out["score"] > -1e8]

        return out.head(top_n)[[
            "course_code", "course_title",
            "Course_department", "Course_slot", "score"
        ]]


# ========= DEMO =========
def demo():
    model = SentenceTransformer(SBERT_MODEL_NAME)

    df = load_processed_courses(model)
    rec = SemanticCourseRecommender(df)

    user_profile = {
        "User_course_interest": ["Machine Learning","Data Science","NLP"],
        "User_career_interest": "ML Engineer",
        "User_previous_courses": ["COL216"],
        "User_occupiedslots": ["A"]
    }

    recs = rec.recommend(user_profile, 10, user_profile["User_occupiedslots"])
    print("\n=== RECOMMENDATIONS ===")
    print(recs.to_string(index=False))


if __name__ == "__main__":
    demo()
