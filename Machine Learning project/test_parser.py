import re
import json

def parse_course_catalog(text):
    """
    Parses the provided course catalog text into structured data.

    Args:
        text (str): The raw text block containing course information.

    Returns:
        list: A list of dictionaries, where each dictionary represents a course.
    """
    
    # 1. Clean up and standardize the text for easier parsing
    # Remove excessive blank lines and leading/trailing whitespace
    cleaned_text = text.strip()
    
    # The first line is the Department name
    department_name = cleaned_text.split('\n')[0].strip()
    
    # Remove the department name and any leading whitespace/newlines for the main parsing block
    course_data_text = cleaned_text[len(department_name):].strip()
    
    # 2. Split the main text into individual course blocks
    # We use a non-capturing group `(?=...)` to split the string just *before*
    # the start of a new course code (ApL###) but keep the course code in the resulting list.
    # The pattern finds 'ApL' followed by 3 digits.
    course_blocks = re.split(r'(?=\nApL\d{3})', '\n' + course_data_text)
    
    # The first element is usually empty or whitespace due to the leading newline, so we slice from 1
    course_blocks = [block.strip() for block in course_blocks if block.strip()]

    parsed_courses = []
    
    # 3. Iterate through each course block and extract details
    for block in course_blocks:
        # Define specific regex patterns for components
        
        # Regex to capture Course Code (e.g., ApL100) and Course Title (e.g., Engineering Mechanics)
        # We look for A-Z + a-z + 3 digits, followed by the title up to a newline or Credits/Pre-requisite line.
        title_match = re.search(r'([A-Za-z]+\d{3})\s+([^(\n]+?)\s*\n', block, re.DOTALL)
        
        course_code = title_match.group(1).strip() if title_match else 'N/A'
        course_title = title_match.group(2).strip() if title_match else 'N/A'
        
        # Regex to capture Credits (e.g., 4 Credits (3-1-0))
        credits_match = re.search(r'(\d+ Credits \(\d-\d-\d\))', block)
        credits = credits_match.group(1).strip() if credits_match else 'N/A'
        
        # The description is the rest of the text, stripped of the parts we extracted.
        # This is the trickiest part due to inconsistent formatting. We will clean the block
        # by removing the Code/Title line and the Credits line, and then clean up line breaks.
        description_raw = block
        
        if title_match:
            # Remove the Course Code and Title from the description block
            description_raw = description_raw.replace(title_match.group(0).strip(), '', 1).strip()
        
        if credits_match:
            # Remove the Credits string from the description block
            description_raw = description_raw.replace(credits_match.group(0).strip(), '', 1).strip()
            
        # Clean up description: replace multiple newlines/whitespace with a single space
        # and remove any residual "Pre-requisite(s):" or "Overlaps with:" for a cleaner final description.
        description = re.sub(r'\s+', ' ', description_raw)
        description = description.replace('Pre-requisite(s):', 'Pre-requisite(s):').replace('Overlaps with:', 'Overlaps with:').strip()

        parsed_courses.append({
            'Department': department_name,
            'Course Code': course_code,
            'Course Title': course_title,
            'Credits': credits,
            'Description': description
        })
        
    return parsed_courses

# --- Input Text ---
course_catalog_text = """
Department of Applied Mechanics
ApL100 Engineering Mechanics 
Rotation of a fluid particle, Vorticity and Circulation, Stream Function,
4 Credits (3-1-0) Irrotational flow and Velocity Potential function. DYNAMICS OF AN
IDEAL FLUID: Continuity and Euler’s Equations of Motion, Bernoulli
Kinematics, Statics, Equations of Motion, Rigid body dynamics,
Equation, Applications to Flow Measurement and other real flow
Introduction to variational mechanics.
problems. MECHANICS OF VISCOUS FLOW: Navier Stokes equations,
ApL101 Applied Mathematics in Engineering Applications 
3 Credits (3-0-0)
a pipe, Friction factor, Applications to Pipe Networks. DIMENSIONAL
Ordinary Differential Equation: Second order ODEs, Method of
ANALYSIS: Similarity of motion, Dimensionless numbers, Modeling
Undetermined Coefficients, Variation of Parameters, Strum-Liouville
of fluid flows, Applications. INTEGRAL ANALYSIS: Reynolds Transport
eigenvalue problem, Difference equation. Partial Differential Equation:
Theorem, Control Volume Analysis.
Classification of PDEs, Heat, Wave and Laplace Equations, Separation
Solid Mechanics: State of stress at a point, equations of motion,
of variables to solve PDEs. Fourier Transform: Fourier sine transform,
principal stress, maximum shear stress. Concept of strain, strain
Fourier cosine Transform, Technique for solving ODEs and PDEs.
displacement relations, compatibility conditions, principal strains,
Probability Theory: Axioms of probability, Conditional probability,
transformation of stress/strain tensor, state of plane stress/strain.
Random variable, Uncertainty in engineering system, Discrete and
Constitutive relations, uniaxial tension test, idealized stress-strain
Continuous distributions, Distribution function, Joint probability
diagrams, isotropic linear elastic and elasto-plastic materials. Energy
distribution, Moments, Covariance, Correlation coefficient. Stochastic
Methods. Uniaxial stress and strain analysis of bars, thermal stresses,
Processes: Definition of Stochastic process, Stochastic FE model,
Torsion, Bending, Stability of Equilibrium.
Stationary process, Markov chain, Poisson process.
ApL103 Experimental Methods
4 Credits (3-0-2)
Experimental Analysis: Types of measurements and errors, Overlaps with: APL107, APL105
Relative frequency distribution, Histogram, True value, Precision of Introduction to Fluids and the concept of viscosity, Flow visualization,
measurement, Method of least squares, the curve fitting, General Fluid Statics, Physical laws for a control volume including continuity,
linear regression,Theory of errors, Binomial and Gaussian distribution, momentum and energy equations, Bernoulli equation, Differential
Chi-square test. equations of fluid motion, Navier Stokes equations, vorticity and
Experimental Methods: Principles of Measurement, Basic Elements potential flows, dimensional analysis and similitude, Boundary layer
of a Measuring Device. theory, 1-D compressible flow.
ApL106 Fluid Mechanics
4 Credits (3-1-0)
Pre-requisite(s): APL100
Overlaps with: APL107, APL105
Introduction to Fluids and the concept of viscosity, Flow visualization,
Fluid Statics, Physical laws for a control volume including continuity,
momentum and energy equations, Bernoulli equation, Differential
equations of fluid motion, Navier Stokes equations, vorticity and
potential flows, dimensional analysis and similitude, Boundary layer
theory, 1-D compressible flow.
"""

# --- Execute Parsing and Output Results ---
try:
    results = parse_course_catalog(course_catalog_text)
    
    print("--- Successfully Parsed Course Catalog ---")
    print("\n")
    
    # Print results in a readable, column-like format
    print(f"{'DEPT':<10} | {'CODE':<8} | {'TITLE':<45} | {'CREDITS':<15} | DESCRIPTION START")
    print("-" * 150)
    for course in results:
        desc_start = course['Description'][:80] + '...'
        print(f"{course['Department'].split(' ')[-1]:<10} | {course['Course Code']:<8} | {course['Course Title']:<45} | {course['Credits']:<15} | {desc_start}")

    print("\n\n--- Raw JSON Output (Full Data) ---")
    print(json.dumps(results, indent=2))
    
except Exception as e:
    print(f"An error occurred during parsing: {e}")