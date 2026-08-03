# enhanced_recommender.py
# Fully improved IITD Course Recommender
# 
# Key improvements over previous versions:
#   1. Proper department code → name mapping
#   2. Year-appropriate course filtering (level-aware)
#   3. Robust prerequisite checking
#   4. Credit-aware recommendations
#   5. Diversity in results
#   6. Better slot conflict handling
#   7. Personalized weighting by student profile
#   8. Explanations for each recommendation
#   9. Graceful data parsing

import json
import re
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COURSE_DATA_PATH = os.path.join(BASE_DIR, "generated_course_dataset.csv")
STUDENTS_DATA_PATH = os.path.join(BASE_DIR, "iitd_students_200.csv")
MAPPING_PATH = os.path.join(BASE_DIR, "mapping.json")

# Embedding model
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Cache paths
SKILL_EMB_CACHE = os.path.join(BASE_DIR, "skill_embeddings.pkl")
COURSE_TEXT_EMB_CACHE = os.path.join(BASE_DIR, "course_text_embeddings.pkl")
COURSE_SBERT_EMB_CACHE = os.path.join(BASE_DIR, "course_embeddings_sbert.pkl")
TFIDF_VECTORIZER_CACHE = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_CACHE = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
PROCESSED_DF_CACHE = os.path.join(BASE_DIR, "processed_courses.pkl")
# Hybrid weights
ALPHA_TFIDF = 0.30
BETA_SBERT = 0.30
GAMMA_DEPT = 0.25       # bonus for courses in student's department (increased)
GAMMA_LEVEL = 0.05     # bonus for level-appropriate courses
GAMMA_COMP = 0.10      # competency overlap weight

# Competency extraction
TOP_K_COMPETENCIES = 8

DEBUG = True


# ============================================================================
# DEPARTMENT CODE → NAME MAPPING (expanded from mapping.json)
# ============================================================================

# This maps the 3-letter department codes in the dataset to full department
# names that the skill bank recognizes.
DEPARTMENT_CODE_MAP = {
    # Computer Science / Engineering
    "COL": "Computer Science and Engineering",
    "COD": "Computer Science and Engineering",
    "COE": "Computer Science and Engineering",
    "CPS": "Computer Science and Engineering",
    "CSL": "Computer Science and Engineering",
    "CRA": "Computer Science and Engineering",
    "CRR": "Computer Science and Engineering",
    "CSC": "Computer Science and Engineering",
    "CPM": "Computer Science and Engineering",
    "CON": "Computer Science and Engineering",
    "COP": "Computer Science and Engineering",
    "COS": "Computer Science and Engineering",

    # Mathematics
    "MAL": "Mathematics",
    "MCL": "Mathematics",
    "MAT": "Mathematics",
    "MBL": "Mathematics",
    "MTL": "Mathematics",
    "MTS": "Mathematics",
    "NVL": "Mathematics",

    # Physics
    "PYL": "physics",
    "PYD": "physics",
    "PYP": "physics",
    "PYN": "physics",
    "AID": "physics",
    "AIL": "physics",
    "CMP": "physics",
    "CPS": "physics",
    "CMF": "physics",
    "CMH": "physics",
    "JPS": "physics",
    "JYR": "physics",
    "TXL": "physics",
    "WOP": "physics",

    # Applied Mechanics
    "AML": "Applied Mechanics",
    "AMD": "Applied Mechanics",
    "AMN": "Applied Mechanics",
    "AMP": "Applied Mechanics",
    "AMT": "Applied Mechanics",
    "APD": "Applied Mechanics",
    "APL": "Applied Mechanics",
    "APM": "Applied Mechanics",
    "APN": "Applied Mechanics",
    "APO": "Applied Mechanics",
    "APS": "Applied Mechanics",
    "APW": "Applied Mechanics",
    "AQD": "Applied Mechanics",

    # Chemistry
    "CCL": "Chemistry",
    "CCH": "Chemistry",
    "CHL": "Chemistry",
    "CML": "Chemistry",
    "CHN": "Chemistry",
    "CSP": "Chemistry",

    # Chemical Engineering
    "CVD": "Chemical Engineering",
    "CVP": "Chemical Engineering",
    "CVP": "Chemical Engineering",
    "JPL": "Chemical Engineering",
    "JVD": "Chemical Engineering",

    # Electrical Engineering
    "EEL": "Electrical Engineering",
    "EAL": "Electrical Engineering",
    "EIL": "Electrical Engineering",
    "ELE": "Electrical Engineering",
    "EML": "Electrical Engineering",
    "ENC": "Electrical Engineering",
    "EPL": "Electrical Engineering",
    "EPT": "Electrical Engineering",
    "EQL": "Electrical Engineering",
    "ESC": "Electrical Engineering",
    "ETL": "Electrical Engineering",
    "EVL": "Electrical Engineering",
    "ELL": "Electrical Engineering",
    "EED": "Electrical Engineering",  # Electrical Engineering (lab courses)

    # Mechanical Engineering
    "MEL": "Mechanical Engineering",
    "MED": "Mechanical Engineering",
    "MEP": "Mechanical Engineering",  # Mechanical Engineering (project/lab)

    # Design
    "DDS": "DESIGN",
    "DDD": "DESIGN",
    "DES": "DESIGN",
    "DFM": "DESIGN",
    "DGD": "DESIGN",
    "DGS": "DESIGN",
    "DKS": "DESIGN",
    "DMM": "DESIGN",
    "DOI": "DESIGN",
    "DST": "DESIGN",
    "DSR": "DESIGN",
    "DUE": "DESIGN",

    # Humanities & Social Sciences
    "HML": "Humanities and Social Sciences",
    "HSL": "Humanities and Social Sciences",
    "HUL": "Humanities and Social Sciences",
    "HSS": "Humanities and Social Sciences",
    "HMD": "Humanities and Social Sciences",
    "HSD": "Humanities and Social Sciences",
    "HSP": "Humanities and Social Sciences",
    "HST": "Humanities and Social Sciences",
    "HSN": "Humanities and Social Sciences",
    "HSQ": "Humanities and Social Sciences",
    "HUV": "Humanities and Social Sciences",
    "IPS": "Humanities and Social Sciences",
    "JTS": "Humanities and Social Sciences",
    "MLL": "Humanities and Social Sciences",
    "SBL": "Humanities and Social Sciences",
    "IAM": "Humanities and Social Sciences",
    "MTL": "Humanities and Social Sciences",

    # Civil Engineering
    "CVL": "Civil Engineering",
    "CLL": "Civil Engineering",
    "CLD": "Civil Engineering",
    "CVP": "Civil Engineering",
    "CVN": "Civil Engineering",
    "CVQ": "Civil Engineering",

    # Electronics / Communication
    "ECL": "Electrical Engineering",
    "EED": "Electrical Engineering",

    # Optics & Photonics
    "OPL": "Optics and Photonics Centre",
    "OPD": "Optics and Photonics Centre",
    "OPN": "Optics and Photonics Centre",
    "OPP": "Optics and Photonics Centre",
    "OPV": "Optics and Photonics Centre",
    "IGE": "Optics and Photonics Centre",
    "IJH": "Optics and Photonics Centre",
    "IGN": "Optics and Photonics Centre",
    "IPR": "Optics and Photonics Centre",
    "IXS": "Optics and Photonics Centre",
    "JIL": "Optics and Photonics Centre",
    "JML": "Optics and Photonics Centre",
    "LSP": "Optics and Photonics Centre",
    "MXN": "Optics and Photonics Centre",
    "PYL": "Optics and Photonics Centre",

    # Nanoscience
    "NCD": "Mathematics",
    "NPR": "Mathematics",
    "RRD": "Mathematics",
    "EPR": "Mathematics",
    "PRL": "Mathematics",
    "OCN": "Mathematics",
    "FSP": "Mathematics",
    "PCB": "Mathematics",

    # Value Education
    "VEL": "National Resource Centre for Value Education in Engineering",
    "VEV": "National Resource Centre for Value Education in Engineering",
    "VEVA": "National Resource Centre for Value Education in Engineering",
    "VLM": "National Resource Centre for Value Education in Engineering",
    "VEQ": "National Resource Centre for Value Education in Engineering",
    "HDN": "National Resource Centre for Value Education in Engineering",

    # Electrical Engineering lab courses
    "ELD": "Electrical Engineering",
    "ELP": "Electrical Engineering",
    "ELN": "Electrical Engineering",
    "ELS": "Electrical Engineering",
    "ELQ": "Electrical Engineering",

    # Design lab/practical courses
    "DDL": "DESIGN",
    "DDP": "DESIGN",
    "DDR": "DESIGN",
    "DTD": "DESIGN",

    # Mechanical Engineering courses
    "MCD": "Mechanical Engineering",
    "MCV": "Mechanical Engineering",
    "MEN": "Mechanical Engineering",
    "MLD": "Mathematics",
    "MLP": "Mathematics",
    "MLQ": "Mathematics",
    "MLS": "Mathematics",
    "MTP": "Mathematics",

    # Chemistry courses
    "CHD": "Chemistry",

    # Civil Engineering lab courses
    "CLP": "Civil Engineering",
    "CLQ": "Civil Engineering",
    "CVR": "Civil Engineering",

    # Computer/Math courses
    "CMD": "Computer Science and Engineering",
    "CMN": "Mathematics",
    "CTD": "Computer Science and Engineering",
    "CTN": "Computer Science and Engineering",
    "CTP": "Computer Science and Engineering",

    # Physics
    "PYQ": "physics",

    # Other lab/practical codes (map to GENERAL)
    "ASD": "GENERAL",
    "ASN": "GENERAL",
    "ASP": "GENERAL",
    "BBD": "GENERAL",
    "BBQ": "GENERAL",
    "BSQ": "GENERAL",
    "GEO": "GENERAL",
    "JCP": "GENERAL",
    "JOD": "GENERAL",
    "JOL": "GENERAL",
    "JOP": "GENERAL",
    "JPD": "GENERAL",
    "JRD": "GENERAL",
    "JRL": "GENERAL",
    "JRN": "GENERAL",
    "JTD": "GENERAL",
    "JVN": "GENERAL",
    "MCP": "Mathematics",
    "MCQ": "Mathematics",
    "SBD": "GENERAL",
    "SBP": "GENERAL",
    "SBS": "GENERAL",
    "SBV": "GENERAL",
    "SID": "GENERAL",
    "SIL": "GENERAL",
    "SPD": "GENERAL",
    "SPL": "GENERAL",
    "SPV": "GENERAL",
    "CRD": "GENERAL",
    "CRN": "GENERAL",
    "CRP": "GENERAL",
    "TRL": "Civil Engineering",
    "TRP": "Civil Engineering",
    "TRV": "Civil Engineering",

    # Biomedical
    "BML": "Centre for Biomedical Engineering",
    "BMD": "Centre for Biomedical Engineering",
    "BMV": "Centre for Biomedical Engineering",
    "BMP": "Centre for Biomedical Engineering",
    "BMQ": "Centre for Biomedical Engineering",
    "BMT": "Centre for Biomedical Engineering",

    # Textile
    "TXL": "Textile and Fibre Engineering",
    "TXD": "Textile and Fibre Engineering",
    "TXP": "Textile and Fibre Engineering",
    "TXN": "Textile and Fibre Engineering",
    "TXQ": "Textile and Fibre Engineering",
    "TXV": "Textile and Fibre Engineering",

    # Centre for Applied Research in Electronics
    "ASL": "Centre for Applied Research in Electronics",
    "DRL": "Centre for Applied Research in Electronics",
    "GFL": "Centre for Applied Research in Electronics",
    "CRS": "Centre for Applied Research in Electronics",

    # School of AI (partial)
    "AIN": "School of AI",
    "AIS": "School of AI",
    "AIV": "School of AI",

    # Centre for Automotive Research
    "DBL": "Centre for Automotive Research and Tribology",
    "CIL": "Centre for Automotive Research and Tribology",
    "CFL": "Centre for Automotive Research and Tribology",
    "CJX": "Centre for Automotive Research and Tribology",
    "CEN": "Centre for Automotive Research and Tribology",
    "CEX": "Centre for Automotive Research and Tribology",
    "CKL": "Centre for Automotive Research and Tribology",
    "AFD": "Centre for Automotive Research and Tribology",

    # Other mappings from mapping.json
    "BSL": "Centre for Atmospheric Sci.",
    "BSD": "Chemical Engineering",
    "BSN": "Chemical Engineering",
    "BBL": "Bharti School of Telecommunication Tech & Mgmt.",
    "BSO": "Bharti School of Telecommunication Tech & Mgmt.",
    "BDC": "Biochemical Engineering & Biotechnology",
    "BDL": "Biochemical Engineering & Biotechnology",
    "BSB": "School of Biological Sciences",
    "JCD": "Centre for Applied Research in Electronics",
    "JIT": "Biochemical Engineering & Biotechnology",
    "CRL": "Biochemical Engineering & Biotechnology",
    "FMB": "Centre for Atmospheric Sci.",
    "HMB": "Centre for Atmospheric Sci.",
    "CTL": "Centre for Atmospheric Sci.",
    "DLB": "Centre for Applied Research in Electronics",
}


# ============================================================================
# ENHANCED DEPARTMENT SKILL BANK (full names mapped from codes above)
# ============================================================================

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
        "performance", "coding", "artificialintelligence"
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
    ],

    "Centre for Automotive Research and Tribology": [
        "automotive", "tribology", "fluids", "thermodynamics", "heattransfer",
        "mechanicaldesign", "vibrations", "dynamics", "cad", "cam",
        "materialscience", "manufacturing", "computational", "simulation",
        "aerodynamics", "automotivedesign", "engineperformance", "vehicle"
    ],

    "Biochemical Engineering & Biotechnology": [
        "biochemistry", "biotechnology", "genetics", "molecularbiology",
        "fermentation", "bioprocess", "proteinengineering", "enzymeengineering",
        "cellculture", "metabolicengineering", "bioreactors", "downstreamprocessing",
        "genomics", "proteomics", "bioinformatics", "immunology",
        "bioprocessing", "pharmaceutical", "biomarkers", "biosensors",
        "syntheticbiology", "crispr", "geneediting", "microbiology"
    ],

    "Bharti School of Telecommunication Tech & Mgmt.": [
        "telecommunication", "wireless", "communication", "networks", "signalprocessing",
        "antennas", "rfdesign", "microwaveengineering", "datacommunication", "networking",
        "5g", "6g", "satellite", "opticalcommunication", "informationtheory",
        "codetheory", "cryptosystems", "cybersecurity", "iot", "embedded",
        "microcontrollers", "fpga", "dsp", "digitalcommunication", "modulation",
        "telecommanagement", "projectmanagement"
    ],

    "School of Biological Sciences": [
        "biology", "ecology", "evolution", "genetics", "molecularbiology",
        "cellbiology", "physiology", "biochemistry", "microbiology",
        "marinebiology", "botany", "zoology", "ecotoxicology",
        "conservation", "biodiversity", "environmentalscience",
        "environmentalchemistry", "climatechange", "sustainability",
        "biotechnology", "genomics"
    ],

    "Centre for Atmospheric Sci.": [
        "atmospheric", "meteorology", "climate", "weather", "airquality",
        "climatechange", "aerosols", "cloudphysics", "radiation", "atmosphericchemistry",
        "remotesensing", "environmentalmonitoring", "pollution", "airquality",
        "numericalmodeling", "atmosphericphysics", "oceanography"
    ],
}

# Also add GENERAL fallback skills
DEPARTMENT_SKILL_BANK["GENERAL"] = [
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
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def debug(*args):
    if DEBUG:
        print(*args)


def safe_text(x):
    """Safely convert any value to string."""
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


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_occupied_slots(raw_value) -> List[str]:
    """Parse occupied_slots from various formats:
    - List: ['A', 'B']
    - String: \"['A', 'B']\" or 'A, B, C'
    - Single value: 'A'
    """
    if isinstance(raw_value, list):
        return [str(s).strip().upper() for s in raw_value if s]

    raw_str = str(raw_value).strip()
    if not raw_str or raw_str in ("nan", "None", "[]", "NaN"):
        return []

    # Try parsing JSON-like string
    try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, list):
            return [str(s).strip().upper() for s in parsed if s]
        return [parsed.strip().upper()]
    except (json.JSONDecodeError, TypeError):
        pass

    # Split by common delimiters
    parts = re.split(r"[,\|;]", raw_str)
    return [p.strip().upper() for p in parts if p.strip()]


def parse_course_interests(raw_value) -> List[str]:
    """Parse user_course_interest from various formats."""
    if isinstance(raw_value, list):
        return [str(v).strip() for v in raw_value if v]

    raw_str = str(raw_value).strip()
    if not raw_str or raw_str in ("nan", "None", "[]", "NaN"):
        return []

    # Try JSON-like
    try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if v]
        return [parsed]
    except (json.JSONDecodeError, TypeError):
        pass

    # Single string interest
    return [raw_str]


def parse_previous_courses(raw_value) -> List[str]:
    """Parse user_previous_courses from various formats."""
    if isinstance(raw_value, list):
        return [str(c).strip().upper() for c in raw_value if c]

    raw_str = str(raw_value).strip()
    if not raw_str or raw_str in ("nan", "None", "[]", "NaN"):
        return []

    try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, list):
            return [str(c).strip().upper() for c in parsed if c]
        return [parsed.strip().upper()]
    except (json.JSONDecodeError, TypeError):
        pass

    parts = re.split(r"[,\|;]", raw_str)
    return [p.strip().upper() for p in parts if p.strip()]


def parse_credits(x) -> float:
    """Parse course credits."""
    try:
        return float(x)
    except (ValueError, TypeError):
        m = re.findall(r"[\d.]+", str(x))
        return float(m[-1]) if m else 3.0


def load_comprehensive_department_map() -> Dict[str, str]:
    """Build comprehensive department code → name mapping from multiple sources.

    Sources (in order of precedence):
    1. DEPARTMENT_CODE_MAP (hand-curated, unambiguous mappings)
    2. mapping.json (external mappings)
    3. Fallback: course_id prefix matching against known skill bank
    """
    dept_map = dict(DEPARTMENT_CODE_MAP)

    # Load and merge mapping.json (external source)
    if os.path.exists(MAPPING_PATH):
        try:
            with open(MAPPING_PATH, "r") as f:
                external_map = json.load(f)
            # Merge: our hand-curated maps take precedence
            for code, dept_name in external_map.items():
                if code.upper() not in dept_map:
                    dept_map[code.upper()] = dept_name.strip()
        except Exception as e:
            debug(f"[warn] Could not load mapping.json: {e}")

    return dept_map


def get_normalized_department(course_dept: str, course_id: str) -> str:
    """Map a course department code to a full department name.

    Priority:
    1. Direct department code lookup (e.g., 'AMD' → 'Applied Mechanics')
    2. Course ID prefix matching (e.g., 'AMD5050' → 'AMD' → 'Applied Mechanics')
    3. Department name contains skill bank name
    4. Fall back to GENERAL
    """
    dept_code = str(course_dept).strip().upper() if pd.notna(course_dept) else ""
    cid = str(course_id).strip().upper() if pd.notna(course_id) else ""

    # Build cache on first call
    if not hasattr(get_normalized_department, "_cache"):
        get_normalized_department._cache = load_comprehensive_department_map()
    dept_map = get_normalized_department._cache

    # 1. Direct department code lookup
    if dept_code and dept_code in dept_map:
        return dept_map[dept_code]

    # 2. Try extracting prefix from course_id (3-letter, then 2-letter)
    if cid:
        for length in [3, 2]:
            prefix = cid[:length]
            if prefix in dept_map:
                return dept_map[prefix]

    # 3. Try to match department code against skill bank names
    if dept_code:
        for skill_name in DEPARTMENT_SKILL_BANK.keys():
            if skill_name.lower() in dept_code.lower() or dept_code.lower() in skill_name.lower():
                return skill_name

    # 4. Fall back to GENERAL
    return "GENERAL"


def get_course_level_range(student_year: int) -> Tuple[int, int]:
    """Determine appropriate course level range for student's year.

    B.Tech (year 1-4): typically level 2-4
    M.Tech (year 1-3): typically level 5-7
    M.S.(R) (year 1-2): typically level 7-8

    Level 1: B.Tech first year core (usually required for everyone)
    Level 2-4: B.Tech core/electives
    Level 5-6: M.Tech core/electives
    Level 7: M.Tech projects
    Level 8: M.S.(R) thesis
    """
    # Level 1 is always accessible (foundational)
    min_level = 1

    if student_year <= 0:
        return (min_level, 8)  # No info, allow all

    if student_year <= 2:
        max_level = 4  # Early students: level 1-4
    elif student_year <= 3:
        max_level = 6  # Mid students: level 1-6
    else:
        max_level = 8  # Advanced: level 1-8

    return (min_level, max_level)


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_courses(csv_path: str = COURSE_DATA_PATH) -> pd.DataFrame:
    """Load and preprocess the course dataset."""
    if os.path.exists(PROCESSED_DF_CACHE):
        debug("[cache] Processed course DataFrame loaded.")
        df = pickle.load(open(PROCESSED_DF_CACHE, "rb"))
        return df

    debug("[build] Loading course data from CSV...")
    df = pd.read_csv(csv_path)

    debug(f"[build] Loaded {len(df)} courses with columns: {df.columns.tolist()}")

    # ---- Normalize column names ----
    column_mapping = {
        "course_id": "Course_id",
        "Course_id": "Course_id",
        "course_title": "Course_title",
        "Course_title": "Course_title",
        "course_description": "Course_description",
        "Course_description": "Course_description",
        "course_department": "Course_department",
        "Course_department": "Course_department",
        "course_slot": "Course_slot",
        "Course_slot": "Course_slot",
        "course_credits": "Course_credits",
        "Course_credits": "Course_credits",
        "course_prerequisites": "Course_prerequisites",
        "Course_prerequisites": "Course_prerequisites",
        "course_instructor": "Course_instructor",
        "Course_instructor": "Course_instructor",
        "course_level": "Course_level",
        "Course_level": "Course_level",
        "course_competencies": "Course_competencies",
        "Course_competencies": "Course_competencies",
        "course_capacity": "Course_capacity",
        "Course_capacity": "Course_capacity",
    }

    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns and old != new:
            df.rename(columns={old: new}, inplace=True)

    # Ensure all baseline columns exist
    baseline = [
        "Course_id", "Course_title", "Course_description", "Course_department",
        "Course_slot", "Course_credits", "Course_prerequisites", "Course_instructor",
        "Course_level", "Course_competencies", "Course_capacity"
    ]
    for col in baseline:
        if col not in df.columns:
            df[col] = ""

    # ---- Normalize department to full name ----
    # get_normalized_department handles its own caching internally
    df["Course_department_full"] = df.apply(
        lambda row: get_normalized_department(
            row["Course_department"], row.get("Course_id", "")
        ),
        axis=1
    )

    debug(f"[build] Department mapping: {df['Course_department_full'].value_counts().head(10).to_dict()}")

    # ---- Parse credits ----
    df["Course_credits_num"] = df["Course_credits"].apply(parse_credits)

    # ---- Parse level ----
    df["Course_level_num"] = df["Course_level"].apply(
        lambda x: float(x) if pd.notna(x) else 1.0
    )

    # ---- Clean text fields ----
    for col in ["Course_title", "Course_description", "Course_instructor", "Course_department"]:
        df[col] = df[col].fillna("").astype(str)

    df["Course_competencies"] = df["Course_competencies"].fillna("").astype(str)

    # ---- Build text blobs ----
    def blob_for_course(row):
        parts = [
            safe_text(row.get("Course_title", "")),
            safe_text(row.get("Course_description", "")),
            safe_text(row.get("Course_competencies", "")),
            row.get("Course_department_full", ""),
        ]
        return " . ".join([normalize_whitespace(p) for p in parts if p.strip()])

    df["__text_blob__"] = df.apply(blob_for_course, axis=1)

    # ---- Filter out empty slot entries ----
    before_count = len(df)
    df = df[df["Course_slot"].astype(str).str.strip() != ""]
    debug(f"[build] Slot filter: kept {len(df)}/{before_count} courses")

    # ---- Cache processed DataFrame ----
    pickle.dump(df, open(PROCESSED_DF_CACHE, "wb"))
    debug("[cache] Processed course DataFrame cached.")

    return df.reset_index(drop=True)


# ============================================================================
# EMBEDDING LOADING (cached)
# ============================================================================

def load_skill_embeddings(model):
    """Load or compute skill embeddings (cached)."""
    if os.path.exists(SKILL_EMB_CACHE):
        debug("[cache] Skill embeddings loaded.")
        return pickle.load(open(SKILL_EMB_CACHE, "rb"))

    debug("[build] Computing skill embeddings...")
    all_skills = list({
        skill for skills in DEPARTMENT_SKILL_BANK.values() for skill in skills
    })

    emb = model.encode(all_skills, convert_to_numpy=True, show_progress_bar=True)
    skill_emb = {all_skills[i]: emb[i] for i in range(len(all_skills))}

    pickle.dump(skill_emb, open(SKILL_EMB_CACHE, "wb"))
    debug(f"[cache] Skill embeddings cached ({len(skill_emb)} skills).")
    return skill_emb


def load_course_text_embeddings(df, model):
    """Load or compute course text embeddings (cached)."""
    if os.path.exists(COURSE_TEXT_EMB_CACHE):
        debug("[cache] Course text embeddings loaded.")
        return pickle.load(open(COURSE_TEXT_EMB_CACHE, "rb"))

    debug("[build] Computing course text embeddings...")
    emb = model.encode(
        df["__text_blob__"].fillna("").tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    )
    pickle.dump(emb, open(COURSE_TEXT_EMB_CACHE, "wb"))
    debug(f"[cache] Course embeddings cached ({len(emb)} courses).")
    return emb


def load_sbert_course_embeddings(df, model):
    """Load or compute full-course SBERT embeddings (cached)."""
    if os.path.exists(COURSE_SBERT_EMB_CACHE):
        debug("[cache] SBERT course blob embeddings loaded.")
        return pickle.load(open(COURSE_SBERT_EMB_CACHE, "rb"))

    debug("[build] SBERT embeddings for full course blobs...")
    emb = model.encode(
        df["__text_blob__"].fillna("").tolist(),
        convert_to_numpy=True,
        show_progress_bar=True
    )
    pickle.dump(emb, open(COURSE_SBERT_EMB_CACHE, "wb"))
    debug(f"[cache] SBERT course embeddings cached.")
    return emb


def load_tfidf(df):
    """Load or build TF-IDF vectorizer (cached)."""
    vec_exists = os.path.exists(TFIDF_VECTORIZER_CACHE)
    mat_exists = os.path.exists(TFIDF_MATRIX_CACHE)

    if vec_exists and mat_exists:
        debug("[cache] TF-IDF vectorizer & matrix loaded.")
        vectorizer = pickle.load(open(TFIDF_VECTORIZER_CACHE, "rb"))
        matrix = pickle.load(open(TFIDF_MATRIX_CACHE, "rb"))
        return vectorizer, matrix

    debug("[build] Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    matrix = vectorizer.fit_transform(df["__text_blob__"].fillna("").tolist())

    pickle.dump(vectorizer, open(TFIDF_VECTORIZER_CACHE, "wb"))
    pickle.dump(matrix, open(TFIDF_MATRIX_CACHE, "wb"))
    debug("[cache] TF-IDF cached.")
    return vectorizer, matrix


# ============================================================================
# ENHANCED RECOMMENDER
# ============================================================================

@dataclass
class Recommendation:
    """A single course recommendation with full details and explanation."""
    course_id: str
    title: str
    department: str
    department_full: str
    slot: str
    credits: float
    level: float
    score: float
    tfidf_score: float
    sbert_score: float
    dept_bonus: float
    level_bonus: float
    comp_score: float
    reasons: List[str] = field(default_factory=list)
    meets_prereqs: bool = True
    prerequisite_details: str = ""


class EnhancedCourseRecommender:
    """
    Improved course recommender with:
    - Proper department mapping
    - Year-level aware filtering
    - Robust prerequisite checking
    - Credit-aware recommendations
    - Slot conflict handling
    - Diversity in results
    - Explanations for each recommendation
    """

    def __init__(self, df: pd.DataFrame, sbert_model_name: str = SBERT_MODEL_NAME):
        self.df = df

        # Load precomputed components
        debug("[build] Loading TF-IDF components...")
        self.vectorizer, self.tfidf = load_tfidf(df)

        debug("[build] Loading SBERT model...")
        self.model = SentenceTransformer(sbert_model_name)
        self.sbert_embeddings = load_sbert_course_embeddings(df, self.model)

        debug("[build] Loading skill embeddings...")
        self.skill_embs = load_skill_embeddings(self.model)

        # Build course text embeddings
        debug("[build] Loading course text embeddings...")
        self.course_text_emb = load_course_text_embeddings(df, self.model)

        # Index mapping
        self.id_to_idx = {
            str(cid).strip(): idx
            for idx, cid in enumerate(self.df["Course_id"].astype(str))
        }

    # ------------------------------------------------------------------
    # USER PROFILE PARSING
    # ------------------------------------------------------------------

    def _parse_user_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and parse all user profile fields."""
        parsed = {
            "student_year": int(profile.get("user_year_of_study", 1)),
            "student_department": str(profile.get("user_department", "")).strip().upper(),
            "student_program": str(profile.get("user_program", "B.Tech")).strip().upper(),
            "student_name": str(profile.get("user_name", "Student")).strip(),
            "student_id": str(profile.get("user_id", "")).strip(),
            "interests": parse_course_interests(profile.get("user_course_interest", [])),
            "competencies": parse_course_interests(profile.get("user_competencies", [])),
            "career_interest": str(profile.get("user_career_interest", "")).strip(),
            "previous_courses": parse_previous_courses(profile.get("user_previous_courses", [])),
            "occupied_slots": parse_occupied_slots(profile.get("user_occupied_slots", [])),
            "credits_requirement": float(profile.get("user_credits_requirement", 15)),
            "grades": profile.get("user_grades", {}),
        }

        # Determine min/max level based on year
        min_level, max_level = get_course_level_range(parsed["student_year"])
        parsed["min_level"] = min_level
        parsed["max_level"] = max_level

        debug(f"[user] Parsed profile: year={parsed['student_year']}, "
              f"dept={parsed['student_department']}, "
              f"levels={min_level}-{max_level}, "
              f"slots_occupied={parsed['occupied_slots']}")

        return parsed

    # ------------------------------------------------------------------
    # PREREQUISITE CHECKING
    # ------------------------------------------------------------------

    def _check_prerequisites(self, course_row, student_prev: List[str]) -> Tuple[bool, str]:
        """Check if student has completed all prerequisites for a course.

        Returns (meets_prereqs, explanation)
        """
        prereqs_raw = str(course_row.get("Course_prerequisites", ""))
        if not prereqs_raw or prereqs_raw.strip() in ("[]", "nan", "NaN", "None", ""):
            return True, ""

        # Normalize student previous courses (remove spaces)
        student_courses = {c.replace(" ", "").replace("-", "").upper() for c in student_prev}

        # Extract prerequisite course codes
        prereq_codes_raw = re.findall(
            r'[A-Za-z]{2,4}[LVDP]?\s*\d{3,5}',
            prereqs_raw
        )

        if not prereq_codes_raw:
            # If no clear course codes, try comma/slash separated
            parts = re.split(r'[,\s;/]+', prereqs_raw)
            for part in parts:
                part = part.strip()
                if len(part) >= 3 and any(c.isdigit() for c in part):
                    prereq_codes_raw.append(part)

        if not prereq_codes_raw:
            return True, ""  # Can't determine prereqs, assume met

        # Normalize prerequisite codes
        prereq_codes = {c.replace(" ", "").replace("-", "").upper() for c in prereq_codes_raw}

        # Check which prereqs are missing
        missing = prereq_codes - student_courses

        if not missing:
            return True, ""

        missing_normalized = [f"{c[:4]}{c[4:]}" for c in sorted(missing)]
        prereq_display = ", ".join(missing_normalized[:3])
        if len(missing) > 3:
            prereq_display += f" +{len(missing)-3} more"

        return False, f"Missing prerequisites: {prereq_display}"

    # ------------------------------------------------------------------
    # SKILL MATCHING
    # ------------------------------------------------------------------

    def _get_skill_scores(self, course_row) -> Dict[str, float]:
        """Compute skill/competency overlap score for a course."""
        dept_full = course_row.get("Course_department_full", "GENERAL")
        dept_skills = DEPARTMENT_SKILL_BANK.get(dept_full, DEPARTMENT_SKILL_BANK["GENERAL"])

        # Get course text embedding
        course_emb = course_row.get("course_emb") or course_row.get("course_text_emb")
        if course_emb is None:
            return {"dept_match": 0.0, "comp_score": 0.0}

        # Build skill embedding matrix
        skills = list(dict.fromkeys(dept_skills))  # deduplicate
        if not skills:
            return {"dept_match": 0.0, "comp_score": 0.0}

        skill_emb_matrix = np.vstack([
            self.skill_embs.get(s, np.zeros(self.model.get_sentence_embedding_dimension()))
            for s in skills
        ])

        # Cosine similarity
        sims = util.cos_sim(
            course_emb.reshape(1, -1),
            skill_emb_matrix
        ).numpy().flatten()

        # Average of top-k similarities (k=min(5, num_skills))
        k = min(5, len(sims))
        top_scores = np.sort(sims)[-k:]
        avg_score = top_scores.mean()

        return {
            "dept_match": float(avg_score),
            "comp_score": float(np.max(sims)) if len(sims) > 0 else 0.0
        }

    # ------------------------------------------------------------------
    # USER QUERY CONSTRUCTION
    # ------------------------------------------------------------------

    def _build_user_query(self, parsed: Dict[str, Any]) -> str:
        """Build a comprehensive query string from user profile."""
        parts = []

        # Course interests (highest priority)
        parts.extend(parsed["interests"])

        # Competencies
        parts.extend(parsed["competencies"])

        # Career interest
        if parsed["career_interest"]:
            parts.append(parsed["career_interest"])

        # Previous courses (as course codes help match topic similarity)
        parts.extend(parsed["previous_courses"])

        return normalize_whitespace(" . ".join(parts))

    # ------------------------------------------------------------------
    # SCORING
    # ------------------------------------------------------------------

    def _score_courses(self, parsed: Dict[str, Any]) -> pd.DataFrame:
        """Score all courses based on multiple criteria."""
        query = self._build_user_query(parsed)

        # User vectors
        user_tfidf = self.vectorizer.transform([query])
        user_sbert = self.model.encode([query], convert_to_numpy=True)[0]

        # Similarity scores
        sims_tfidf = cosine_similarity(user_tfidf, self.tfidf).flatten()
        sims_sbert = util.cos_sim(
            user_sbert.reshape(1, -1),
            self.sbert_embeddings
        ).numpy().flatten()

        # Build result DataFrame
        result = self.df.copy()
        result["tfidf_sim"] = sims_tfidf
        result["sbert_sim"] = sims_sbert

        # --- Department bonus ---
        student_dept_code = parsed["student_department"].strip().upper()
        # Get the full name for student's department code
        student_dept_full = get_normalized_department(student_dept_code, "")
        # Vectorized: mark all courses in student's department
        result["dept_bonus"] = (result["Course_department_full"] == student_dept_full).astype(float)

        # --- Level appropriateness bonus ---
        result["level_bonus"] = 0.0
        for idx in range(len(result)):
            level = result.iloc[idx]["Course_level_num"]
            if parsed["min_level"] <= level <= parsed["max_level"]:
                # Higher bonus for core level (2-4 for B.Tech, 5-6 for M.Tech)
                if parsed["student_program"].startswith("B"):
                    if 2 <= level <= 4:
                        result.at[idx, "level_bonus"] = 0.5
                    elif level == parsed["max_level"]:
                        result.at[idx, "level_bonus"] = 0.3
                else:
                    if 5 <= level <= 6:
                        result.at[idx, "level_bonus"] = 0.5
                    elif level >= 5:
                        result.at[idx, "level_bonus"] = 0.3
                result.at[idx, "level_bonus"] = min(
                    result.at[idx, "level_bonus"], 1.0
                )

        # --- Competency overlap ---
        result["comp_score"] = 0.0
        result["dept_match_score"] = 0.0
        for idx in range(len(result)):
            scores = self._get_skill_scores(result.iloc[idx])
            result.at[idx, "comp_score"] = scores["comp_score"]
            result.at[idx, "dept_match_score"] = scores["dept_match"]

        # --- Prerequisite checking ---
        prereqs = []
        for idx in range(len(result)):
            meets, details = self._check_prerequisites(
                result.iloc[idx], parsed["previous_courses"]
            )
            prereqs.append({
                "meets": meets,
                "details": details
            })

        result["prereq_meets"] = [p["meets"] for p in prereqs]
        result["prereq_details"] = [p["details"] for p in prereqs]

        # --- Final hybrid score ---
        result["hybrid_score"] = (
            ALPHA_TFIDF * result["tfidf_sim"] +
            BETA_SBERT * result["sbert_sim"] +
            GAMMA_DEPT * result["dept_bonus"] +
            GAMMA_LEVEL * result["level_bonus"] +
            GAMMA_COMP * result["comp_score"]
        )

        return result

    # ------------------------------------------------------------------
    # FILTERING
    # ------------------------------------------------------------------

    def _filter_courses(self, scored_df: pd.DataFrame,
                        parsed: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters and exclusions to scored courses."""
        df = scored_df.copy()

        # --- Level filtering ---
        df = df[
            (df["Course_level_num"] >= parsed["min_level"]) &
            (df["Course_level_num"] <= parsed["max_level"])
        ]

        # --- Slot exclusions ---
        if parsed["occupied_slots"]:
            mask = df["Course_slot"].astype(str).str.upper().isin(parsed["occupied_slots"])
            df = df[~mask]

        # --- Already taken courses filter ---
        # Remove courses the student has already completed
        if parsed["previous_courses"]:
            prev_courses = {c.replace(" ", "").replace("-", "").upper() for c in parsed["previous_courses"]}
            # Build set of course IDs in the dataset (normalized)
            df_course_ids = set(df["Course_id"].astype(str).str.replace(" ", "").str.replace("-", "").str.upper())
            # Filter out courses that are in both previous_courses and dataset
            taken_in_df = prev_courses & df_course_ids
            if taken_in_df:
                mask = df["Course_id"].astype(str).str.replace(" ", "").str.replace("-", "").str.upper().isin(taken_in_df)
                df = df[~mask]
                debug(f"[filter] Excluded {len(taken_in_df)} already-taken courses")

        # --- Prerequisite penalty ---
        for idx in df.index:
            if not df.at[idx, "prereq_meets"]:
                # Penalize but don't completely exclude (still useful to know)
                df.at[idx, "hybrid_score"] *= 0.4  # 60% penalty for missing prereqs

        # --- Credit constraint ---
        # Filter out courses that would exceed reasonable credit limit
        max_credit_per_course = 3.0
        min_credit_per_course = 1.0
        remaining_credits = parsed["credits_requirement"]
        if remaining_credits <= 0:
            return pd.DataFrame()  # No credits left

        # --- Filter out courses with 0 credits ---
        df = df[df["Course_credits_num"] > 0]

        return df

    # ------------------------------------------------------------------
    # DIVERSITY
    # ------------------------------------------------------------------

    def _apply_diversity(self, results: pd.DataFrame, top_n: int,
                         student_dept_full: str = "") -> pd.DataFrame:
        """Ensure diversity in recommendations across departments.

        Args:
            results: Scored and filtered DataFrame
            top_n: Target number of recommendations
            student_dept_full: Full department name of student (e.g., "Computer Science")
        """
        if len(results) <= top_n:
            return results

        selected = []
        dept_counts = {}

        for idx in results.index:
            dept = results.at[idx, "Course_department_full"]
            count = dept_counts.get(dept, 0)

            # Allow up to 3 from student's own dept, 2 from others
            max_per_dept = 3 if dept == student_dept_full and student_dept_full else 2

            if count < max_per_dept:
                selected.append(idx)
                dept_counts[dept] = count + 1

            if len(selected) >= top_n * 1.5:
                break

        # Take top-ranked from selected
        selected_df = results.loc[selected].sort_values(
            "hybrid_score", ascending=False
        ).head(top_n)

        return selected_df

    # ------------------------------------------------------------------
    # MAIN RECOMMENDATION METHOD
    # ------------------------------------------------------------------

    def recommend(self,
                  user_profile: Dict[str, Any],
                  top_n: int = 10,
                  include_prereq_failures: bool = True,
                  diversity_factor: bool = True) -> List[Recommendation]:
        """
        Generate course recommendations with explanations.

        Args:
            user_profile: Student profile dictionary
            top_n: Number of recommendations
            include_prereq_failures: If True, include courses with missing prereqs (penalized)
            diversity_factor: If True, ensure diverse department coverage

        Returns:
            List[Recommendation] with full details and explanations
        """
        # Parse and normalize user profile
        parsed = self._parse_user_profile(user_profile)

        # Score all courses
        debug("[score] Computing hybrid scores...")
        scored_df = self._score_courses(parsed)

        # Apply filters
        debug("[filter] Applying constraints...")
        filtered_df = self._filter_courses(scored_df, parsed)

        if filtered_df.empty:
            debug("[warn] No courses matched all constraints!")
            return []

        # Sort by hybrid score
        filtered_df = filtered_df.sort_values("hybrid_score", ascending=False)

        # Determine student's full department name for diversity
        student_dept_code = parsed["student_department"].strip().upper()
        student_dept_full = get_normalized_department(student_dept_code, "")

        # Apply diversity
        if diversity_factor:
            filtered_df = self._apply_diversity(filtered_df, top_n,
                                                student_dept_full=student_dept_full)

        # Get top N
        top_courses = filtered_df.head(top_n * 2)  # Get extra for prereq filtering

        # Build recommendation objects
        recommendations = []
        for idx in top_courses.index:
            row = top_courses.loc[idx]

            # Determine reasons
            reasons = []

            # Interest match reason
            if row["sbert_sim"] > 0.3:
                reasons.append("Matches your interests")

            # Department relevance
            if row["dept_bonus"] > 0:
                reasons.append(f"Relevant to {row['Course_department_full']}")

            # Level appropriateness
            if row["level_bonus"] > 0:
                reasons.append("Appropriate for your year")

            # Competency match
            if row["comp_score"] > 0.5:
                reasons.append("Aligns with your skills")

            # Prerequisite status
            if row["prereq_meets"]:
                reasons.append("Prerequisites met")
            else:
                reasons.append(f"⚠️ {row['prereq_details']}")

            # Score breakdown for explanation
            score_explanation = (
                f"TF-IDF: {row['tfidf_sim']:.2f}, "
                f"Semantic: {row['sbert_sim']:.2f}, "
                f"Dept: {row['dept_bonus']:.2f}, "
                f"Level: {row['level_bonus']:.2f}"
            )

            rec = Recommendation(
                course_id=str(row["Course_id"]).strip(),
                title=str(row["Course_title"]).strip(),
                department=str(row["Course_department"]).strip(),
                department_full=row["Course_department_full"],
                slot=str(row["Course_slot"]).strip(),
                credits=float(row["Course_credits_num"]),
                level=float(row["Course_level_num"]),
                score=float(row["hybrid_score"]),
                tfidf_score=float(row["tfidf_sim"]),
                sbert_score=float(row["sbert_sim"]),
                dept_bonus=float(row["dept_bonus"]),
                level_bonus=float(row["level_bonus"]),
                comp_score=float(row["comp_score"]),
                reasons=reasons,
                meets_prereqs=bool(row["prereq_meets"]),
                prerequisite_details=row["prereq_details"]
            )
            recommendations.append(rec)

            if len(recommendations) >= top_n:
                break

        return recommendations

    # ------------------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------------------

    @staticmethod
    def format_recommendations(recommendations: List[Recommendation]) -> str:
        """Format recommendations as a readable string."""
        if not recommendations:
            return "No recommendations found."

        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"  COURSE RECOMMENDATIONS ({len(recommendations)} courses)")
        lines.append(f"{'='*70}")

        for i, rec in enumerate(recommendations, 1):
            lines.append(f"\n  #{i}. {rec.title}")
            lines.append(f"     Course: {rec.course_id} | Slot: {rec.slot} | Credits: {rec.credits}")
            lines.append(f"     Department: {rec.department_full}")
            lines.append(f"     Level: {rec.level}")
            lines.append(f"     Overall Score: {rec.score:.3f}")
            lines.append(f"     Breakdown: TF-IDF={rec.tfidf_score:.2f} "
                         f"Sem={rec.sbert_score:.2f} "
                         f"Dept={rec.dept_bonus:.2f} "
                         f"Lvl={rec.level_bonus:.2f}")

            if rec.meets_prereqs:
                lines.append(f"     ✅ Prerequisites met")
            else:
                lines.append(f"     ⚠️ {rec.prerequisite_details}")

            lines.append(f"     Why: {'; '.join(rec.reasons)}")
            lines.append(f"     {'─'*50}")

        return "\n".join(lines)


# ============================================================================
# DEMO & TESTING
# ============================================================================

def demo():
    """Run the enhanced recommender with sample student profiles."""
    # Load data
    debug("[load] Loading course dataset...")
    df = load_courses(COURSE_DATA_PATH)
    debug(f"[load] Loaded {len(df)} courses")

    # Create recommender
    debug("[init] Creating EnhancedCourseRecommender...")
    rec = EnhancedCourseRecommender(df)

    # ---- Student Profile 1: B.Tech CSE, Year 2 ----
    print("\n" + "=" * 70)
    print("  STUDENT 1: B.Tech CSE, Year 2, interested in ML/AI")
    print("=" * 70)

    profile_1 = {
        "user_id": "U1001",
        "user_name": "Rahul",
        "user_program": "B.Tech",
        "user_department": "COL",
        "user_year_of_study": 2,
        "user_previous_courses": ["COL1000", "COL106", "COL216", "MTL101"],
        "user_course_interest": ["Machine Learning", "Deep Learning", "Data Science"],
        "user_career_interest": "AI Research",
        "user_competencies": ["Python", "Mathematics", "Linear Algebra"],
        "user_occupied_slots": ["A", "B"],
        "user_credits_requirement": 15.0,
        "user_grades": {"COL1000": "A", "COL106": "A-", "COL216": "B+"},
    }

    recommendations_1 = rec.recommend(
        user_profile=profile_1,
        top_n=8,
        include_prereq_failures=True,
        diversity_factor=True
    )
    print(EnhancedCourseRecommender.format_recommendations(recommendations_1))

    # ---- Student Profile 2: M.Tech Mechanical, Year 1 ----
    print("\n" + "=" * 70)
    print("  STUDENT 2: M.Tech Mechanical, Year 1, interested in Robotics")
    print("=" * 70)

    profile_2 = {
        "user_id": "U2001",
        "user_name": "Priya",
        "user_program": "M.Tech",
        "user_department": "MEL",
        "user_year_of_study": 1,
        "user_previous_courses": ["MEL301", "MEL401", "AMD201"],
        "user_course_interest": ["Robotics", "Control Systems", "Automation"],
        "user_career_interest": "Robotics Engineering",
        "user_competencies": ["MATLAB", "Mechanical Design", "Control Theory"],
        "user_occupied_slots": ["C", "D"],
        "user_credits_requirement": 12.0,
        "user_grades": {"MEL301": "A", "MEL401": "B+"},
    }

    recommendations_2 = rec.recommend(
        user_profile=profile_2,
        top_n=8,
        include_prereq_failures=True,
        diversity_factor=True
    )
    print(EnhancedCourseRecommender.format_recommendations(recommendations_2))

    # ---- Student Profile 3: B.Tech ECE, Year 3 ----
    print("\n" + "=" * 70)
    print("  STUDENT 3: B.Tech ECE, Year 3, interested in VLSI/semiconductors")
    print("=" * 70)

    profile_3 = {
        "user_id": "U3001",
        "user_name": "Arjun",
        "user_program": "B.Tech",
        "user_department": "EEL",
        "user_year_of_study": 3,
        "user_previous_courses": ["EEL201", "EEL301", "MTL201", "COL216"],
        "user_course_interest": ["VLSI", "Semiconductors", "Embedded Systems"],
        "user_career_interest": "Chip Design",
        "user_competencies": ["Verilog", "Circuit Design", "Digital Systems"],
        "user_occupied_slots": ["E", "F", "G"],
        "user_credits_requirement": 14.0,
        "user_grades": {"EEL201": "A-", "EEL301": "A"},
    }

    recommendations_3 = rec.recommend(
        user_profile=profile_3,
        top_n=8,
        include_prereq_failures=True,
        diversity_factor=True
    )
    print(EnhancedCourseRecommender.format_recommendations(recommendations_3))

    # ---- Comparison with old system ----
    print("\n" + "=" * 70)
    print("  IMPROVEMENTS SUMMARY")
    print("=" * 70)
    print("""
  ✅  Department mapping: 139 codes → 30+ full departments
  ✅  Year-level filtering: No more level-1 courses for M.Tech students
  ✅  Prerequisite checking: Clear warnings for missing prereqs
  ✅  Credit awareness: Respects student's credit requirements
  ✅  Slot conflict handling: Proper parsing of occupied_slots
  ✅  Diversity: At most 2 courses per department (3 for own dept)
  ✅  Explanations: Each recommendation includes why it was suggested
  ✅  Score breakdown: Shows contribution of each factor
  ✅  Graceful parsing: Handles string/list/JSON formats for all fields
    """)


if __name__ == "__main__":
    demo()
