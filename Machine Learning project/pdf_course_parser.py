
"""
pdf_course_parser.py
Robust parser to extract department and course entries from a PDF (text or scanned).
Dependencies:
    pip install pdfplumber regex pytesseract pillow
System dependency (optional OCR): tesseract-ocr (if parsing scanned PDFs)

Functions:
    parse_pdf_courses(pdf_path, use_ocr=False) -> List[dict]
    save_courses_json(courses, out_path)

Example:
    courses = parse_pdf_courses("Course_document.pdf")
    save_courses_json(courses, "courses.json")
"""
import re
import json
from typing import List, Dict, Optional

def _normalize_whitespace(s: str) -> str:
    # collapse many spaces/newlines, but keep paragraph separation
    s = s.replace('\r', '\n')
    # replace multiple spaces with single
    s = re.sub(r'[ \t]+', ' ', s)
    # collapse >2 newlines into two (paragraph sep)
    s = re.sub(r'\n{3,}', '\n\n', s)
    # strip spaces on each line
    s = '\\n'.join([ln.strip() for ln in s.splitlines()])
    # collapse multiple spaces again
    s = re.sub(r' {2,}', ' ', s)
    return s.strip()

def _extract_text_with_pdfplumber(pdf_path: str) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # try to get "layout" text first (pdfplumber default)
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if not text:
                # fallback to extracting chars and join by y0 position (heuristic)
                chars = page.chars
                if chars:
                    # sort by top then left and join
                    chars_sorted = sorted(chars, key=lambda c: (round(c.get('top',0)), round(c.get('x0',0))))
                    lines = []
                    cur_top = None
                    cur_line = []
                    for c in chars_sorted:
                        top = round(c.get('top',0))
                        if cur_top is None:
                            cur_top = top
                        if abs(top - cur_top) > 3:
                            lines.append(''.join([ch.get('text','') for ch in cur_line]))
                            cur_line = [c]
                            cur_top = top
                        else:
                            cur_line.append(c)
                    if cur_line:
                        lines.append(''.join([ch.get('text','') for ch in cur_line]))
                    text = '\\n'.join(lines)
            if text:
                parts.append(text)
    return '\\n\\n'.join(parts)

def _ocr_pdf_pages(pdf_path: str) -> str:
    # Optional OCR fallback for scanned PDFs: requires pytesseract and pillow and tesseract-ocr installed
    try:
        from pdf2image import convert_from_path
        import pytesseract
        images = convert_from_path(pdf_path, dpi=200)
        pages = []
        for im in images:
            pages.append(pytesseract.image_to_string(im))
        return '\\n\\n'.join(pages)
    except Exception as e:
        raise RuntimeError("OCR fallback failed. Install pdf2image, pytesseract and tesseract-ocr or avoid use_ocr=True.") from e

COURSE_CODE_PATTERN = re.compile(r'\\b([A-Z]{2,5}\\d{3})\\b')

CREDITS_PATTERN = re.compile(r'(\\d+\\s*Credits?\\s*\\(\\s*\\d+-\\d+-\\d+\\s*\\))', re.IGNORECASE)
PREREQ_PATTERN = re.compile(r'Pre-?requisite\\(s\\)?:\\s*(.+?)(?=(Overlaps with:|Overlaps|Introduc|Description:|$))', re.IGNORECASE|re.DOTALL)
OVERLAPS_PATTERN = re.compile(r'Overlaps with:\\s*(.+?)(?=(Pre-?requisite|Introduction|$))', re.IGNORECASE|re.DOTALL)

def _split_into_course_chunks(full_text: str):
    """
    Find positions of course-code like tokens and split text into chunks for each course.
    """
    matches = list(COURSE_CODE_PATTERN.finditer(full_text))
    if not matches:
        # fallback: split by a pattern of uppercase title line e.g., "APL100 Engineering Mechanics"
        # return [full_text]
        return [(None, full_text)]

    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(full_text)
        chunks.append((m.group(1), full_text[start:end].strip()))
    return chunks

def parse_course_chunk(code: str, chunk_text: str) -> Dict[str, Optional[str]]:
    """
    Parse a chunk that starts with the given course code.
    Returns dict with Code, Title, Credits, Description, Prerequisites, Overlaps.
    """
    text = _normalize_whitespace(chunk_text)
    # remove the leading code from text if duplicated
    text_after_code = text
    if text_after_code.startswith(code):
        text_after_code = text_after_code[len(code):].lstrip(' -:—')
    # Attempt to find credits
    credits_m = CREDITS_PATTERN.search(text_after_code)
    credits = credits_m.group(1).strip() if credits_m else None

    # Title: first line up to credits or up to newline
    title = None
    # If credits included in same line, try to capture portion before credits
    if credits_m:
        before_credits = text_after_code[:credits_m.start()]
        # take the first non-empty line from before_credits
        lines = [ln for ln in before_credits.splitlines() if ln.strip()]
        if lines:
            title = lines[0].strip(' -:—')
    else:
        # take first line as title
        lines = [ln for ln in text_after_code.splitlines() if ln.strip()]
        if lines:
            # often titles are short; if first line looks like "4 Credits..." then skip
            candidate = lines[0]
            if re.search(r'Credits', candidate, re.IGNORECASE) and len(lines) > 1:
                title = lines[1]
            else:
                # remove stray leading '1.' or 'A.' numbering
                title = re.sub(r'^[\d\.\)\-]+\s*', '', candidate)
    title = title.strip() if title else None

    # Prerequisite
    prereq_m = PREREQ_PATTERN.search(text_after_code)
    prereq = prereq_m.group(1).strip() if prereq_m else None

    overlaps_m = OVERLAPS_PATTERN.search(text_after_code)
    overlaps = overlaps_m.group(1).strip() if overlaps_m else None

    # Description: remove title and credits from chunk, then collapse to paragraph
    desc = text_after_code
    # remove title (first line)
    if title:
        desc = desc.replace(title, '', 1).strip()
    if credits:
        desc = desc.replace(credits, '', 1).strip()
    # remove code occurrences at start
    desc = re.sub(r'^\s*' + re.escape(code), '', desc).strip(' -:—\n')
    # remove leading 'Pre-requisite(s):' and 'Overlaps with:' content from description since extracted separately
    desc = re.sub(r'Pre-?requisite\\(s\\)?:', '', desc, flags=re.IGNORECASE)
    desc = re.sub(r'Overlaps with:', '', desc, flags=re.IGNORECASE)
    # single-line paragraph
    desc = ' '.join([ln.strip() for ln in desc.splitlines() if ln.strip()])
    # truncate repeated course code at end if any
    desc = desc.strip(' .;,')

    return {
        "Department": None,  # filled later
        "Course Code": code,
        "Course Title": title,
        "Credits": credits,
        "Prerequisites": prereq,
        "Overlaps": overlaps,
        "Description": desc or None
    }

def find_department_header(full_text: str) -> Optional[str]:
    # Try to find lines like "Department of Applied Mechanics"
    m = re.search(r'(Department\\s+of\\s+[A-Z][A-Za-z &]+)', full_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: first few lines capitalized
    first_lines = '\\n'.join(full_text.splitlines()[:8])
    m2 = re.search(r'([A-Z][A-Za-z ]{5,30}Department)', first_lines)
    if m2:
        return m2.group(1).strip()
    return None

def parse_pdf_courses(pdf_path: str, use_ocr: bool=False) -> List[Dict]:
    """
    Main entrypoint. Returns a list of course dicts.
    If use_ocr=True, a slower OCR fallback will be attempted (requires pdf2image/pytesseract).
    """
    try:
        full_text = _extract_text_with_pdfplumber(pdf_path)
    except Exception as e:
        # attempt OCR if requested
        if use_ocr:
            full_text = _ocr_pdf_pages(pdf_path)
        else:
            raise

    full_text = _normalize_whitespace(full_text)
    dept = find_department_header(full_text)

    chunks = _split_into_course_chunks(full_text)
    courses = []
    for code, chunk in chunks:
        parsed = parse_course_chunk(code, chunk)
        parsed["Department"] = dept
        courses.append(parsed)
    return courses

def save_courses_json(courses: List[Dict], out_path: str):
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(courses, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse courses from a PDF into JSON.")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--out", default="courses.json", help="Output JSON file")
    parser.add_argument("--ocr", action="store_true", help="Use OCR fallback (slower, requires tesseract)")
    args = parser.parse_args()

    courses = parse_pdf_courses(args.pdf, use_ocr=args.ocr)
    save_courses_json(courses, args.out)
    print(f"Extracted {len(courses)} course entries to {args.out}")
