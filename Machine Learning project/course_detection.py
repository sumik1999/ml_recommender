import re
import json

def parse_courses(filepath):
    department_pattern = re.compile(r"^\s*Department of (.+)", re.I)
    course_start_pattern = re.compile(
        r"^(?P<code>[A-Za-z]{2,3}[LVDP]?\d{2,4})\s+(?P<title>.+)"
    )
    credits_pattern = re.compile(r"(\d+(\.\d+)?)\s*Credits?\s*\((.*?)\)", re.I)

    courses = []
    current_dept = None
    current_course = None
    buffer_desc = []
    line_number = 0

    def flush_current():
        """Finalize the current course entry safely."""
        nonlocal current_course, buffer_desc
        if current_course:
            # Join description lines
            desc = " ".join(buffer_desc).strip()
            current_course["course_description"] = desc if desc else None
            courses.append(current_course)
        current_course = None
        buffer_desc = []

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            line_number += 1

            # Detect department header
            dept_match = department_pattern.match(line)
            if dept_match:
                flush_current()
                current_dept = dept_match.group(1).strip()
                continue

            # Detect start of a new course
            course_match = course_start_pattern.match(line)
            if course_match:
                flush_current()
                code = course_match.group("code").strip()
                title = course_match.group("title").strip()

                current_course = {
                    "department": current_dept,
                    "course_code": code,
                    "course_title": title,
                    "credits": None,
                    "course_description": "",
                    "error_line": None
                }
                continue

            # Extract credits (may appear on same line or separate)
            if current_course:
                credit_match = credits_pattern.search(line)
                if credit_match:
                    current_course["credits"] = credit_match.group(1)
                    continue

            # If neither dept nor course start but still inside course → description
            if current_course:
                if line.strip() != "":
                    buffer_desc.append(line)
            else:
                # Unrecognized content → flag as error
                courses.append({
                    "department": current_dept,
                    "course_code": None,
                    "course_title": None,
                    "course_description": line,
                    "credits": None,
                    "error_line": line_number
                })

    # Flush last course
    flush_current()

    return courses


# ----------- RUN -------------
input_file = "extracted_text.txt"  # Your uploaded file
parsed_output = parse_courses(input_file)

# Save JSON
with open("parsed_courses.json", "w", encoding="utf-8") as out:
    json.dump(parsed_output, out, indent=4)

print("Parsing complete. JSON saved to: parsed_courses.json")
