import re

def clean_chunks(input_file_path, output_file_path):
    """
    Cleans the text chunks by removing website URLs, metadata-like lines,
    handling '##' markers, and filtering out non-content sections.

    Args:
        input_file_path (str): Path to the input chunks.txt file.
        output_file_path (str): Path to the output cleaned_chunks.txt file.
    """
    cleaned_lines = []
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Remove lines containing the website URL
            if "ethiofetena.com" in line:
                continue

            # Filter out lines that seem like book metadata or titles
            if re.match(r"^(general science student textbook|grade \d+|authors :|editors and evaluators :|illustration and layout design|addis ababa city administration education bureau)$", line.lower()):
                continue

            # Filter out lines that are just page numbers or roman numerals
            if re.match(r"^(page number|\d+|i+|ii+|iii+|iv+|v+|vi+|vii+|viii+|ix+|x+)$", line.lower()):
                continue

            # Handle '##' markers within a line (try to merge parts)
            parts = [p.strip() for p in line.split('##')]
            cleaned_line = " ".join(parts).strip()

            cleaned_lines.append(cleaned_line)

    # Further filtering to remove table of contents-like lines and acknowledgements
    final_cleaned_lines = []
    skip_next = 0
    for i, line in enumerate(cleaned_lines):
        if skip_next > 0:
            skip_next -= 1
            continue

        if re.match(r"^table of contents$", line.lower()):
            skip_next = 5 # Skip a few lines after "table of contents"
            continue

        if re.match(r"^acknowledgement$", line.lower()):
            skip_next = 10 # Skip a few lines after "acknowledgement"
            continue

        if re.match(r"^learning outcomes.*$", line.lower()):
            continue # Skip learning outcome headings

        if re.match(r"^main contents.*$", line.lower()):
            skip_next = 3 # Skip a few lines after "main contents"
            continue

        if re.match(r"^review exercise.*$", line.lower()):
            continue

        if re.match(r"^project work.*$", line.lower()):
            continue

        if re.match(r"^key words.*$", line.lower()):
            continue

        if re.match(r"^exercise \d+\.\d+.*$", line.lower()):
            continue

        if re.match(r"^activity \d+\.\d+.*$", line.lower()):
            continue

        if "( m. sc. )" in line.lower() or "( b. sc. )" in line.lower() or "( bed )" in line.lower() or "( med )" in line.lower() or "( ma )" in line.lower():
            continue # Skip author/editor qualifications

        final_cleaned_lines.append(line)

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        outfile.write("\n".join(final_cleaned_lines))

if __name__ == "__main__":
    input_file = "data/chunks.txt"
    output_file = "data/cleaned_chunks.txt"
    clean_chunks(input_file, output_file)
    print(f"Cleaned chunks saved to {output_file}")