import os
from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdf(pdf_path, output_path):
    """Extracts text from a PDF and saves it to a text file."""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write(full_text)
        print(f"Text extracted from {pdf_path} and saved to {output_path}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    pdf_file = "data/raw/grade 7-general science_ethiofetenacom_d837.pdf"  # Adjust if your PDF has a different name
    output_file = "data/chunks.txt"

    # Create the 'data' directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(pdf_file):
        extract_text_from_pdf(pdf_file, output_file)
    else:
        print(f"Error: PDF file not found at {pdf_file}")