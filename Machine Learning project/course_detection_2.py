import re
import json

def parse_courses(filepath):
    department_pattern = re.compile(r"^\s*Department of (.+)", re.I)
    course_start_pattern = re.compile(
        r"^(?P<code>[A-Za-z]{2,3}[LVDP]?\d{2,4})\s+(?P<title>.+)"
    )
    credits_pattern = re.compile(r"(\d+(\.\d+)?)\s*Credits?\s*\((.*?)\)", re.I)

    # Capture prerequisites:
    # Examples:
    # Pre-requisite(s): APL100/APL104
    # Prerequisite(s): APL100, APL104 or equivalent
    prereq_line_pattern = re.compile(
        r"Pre[- ]?requisite\(s\):\s*(.+)", re.I
    )

    # Capture individual course codes (like APL100, MTL107)
    course_code_pattern = re.compile(r"[A-Za-z]{2,3}[LVDP]?\d{2,4}")

    courses = []
    current_dept = None
    current_course = None
    buffer_desc = []
    line_number = 0

    def flush_current():
        """Finalize and store current course safely."""
        nonlocal current_course, buffer_desc
        if current_course:
            desc = " ".join(buffer_desc).strip()
            current_course["course_description"] = desc if desc else None
            courses.append(current_course)
        current_course = None
        buffer_desc = []

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line_number += 1
            line = raw_line.strip()

            # 1. Detect department header
            dept_match = department_pattern.match(line)
            if dept_match:
                flush_current()
                current_dept = dept_match.group(1).strip()
                continue

            # 2. Detect start of a new course
            course_match = course_start_pattern.match(line)
            if course_match:
                flush_current()
                current_course = {
                    "department": current_dept,
                    "course_code": course_match.group("code").strip(),
                    "course_title": course_match.group("title").strip(),
                    "credits": None,
                    "prerequisites": [],
                    "course_description": "",
                    "error_line": None
                }
                continue

            # 3. Detect credits
            if current_course:
                credit_match = credits_pattern.search(line)
                if credit_match:
                    current_course["credits"] = credit_match.group(1)
                    continue

            # 4. Detect prerequisites line
            if current_course:
                prereq_line_match = prereq_line_pattern.search(line)
                if prereq_line_match:
                    raw_prereq = prereq_line_match.group(1)

                    # Extract ALL course codes inside the prerequisite string
                    found_codes = course_code_pattern.findall(raw_prereq)
                    current_course["prerequisites"] = list(set(found_codes))
                    # continue processing because prerequisites lines also appear above descriptions
                    continue

            # 5. Collect course description lines
            if current_course:
                if line != "":
                    buffer_desc.append(line)
            else:
                # 6. Unrecognized content outside any course → flag
                courses.append({
                    "department": current_dept,
                    "course_code": None,
                    "course_title": None,
                    "credits": None,
                    "prerequisites": None,
                    "course_description": line,
                    "error_line": line_number
                })

    # Final flush
    flush_current()

    return courses


# ----------- RUN PARSER ----------
input_file = "extracted_text.txt"   # your file path
parsed_output = parse_courses(input_file)

# Save JSON
with open("parsed_courses2.json", "w", encoding="utf-8") as out:
    json.dump(parsed_output, out, indent=4)

print("Parsing complete. JSON saved to parsed_courses2.json")
