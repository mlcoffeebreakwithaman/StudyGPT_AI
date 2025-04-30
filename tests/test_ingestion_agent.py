# test_ingestion_agent.py (to be placed in the tests/ directory)

import unittest
import os
from core.agents.ingestion_agent import IngestionAgent
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Create a dummy PDF file for testing
DUMMY_PDF_PATH = "data/raw/dummy.pdf"
TOKENIZER = SentenceTransformer('all-MiniLM-L6-v2').tokenizer
EMBEDDING_DIMENSION = 384
FAISS_INDEX_PATH = "data/index_test.faiss" # Separate index for testing

def create_dummy_pdf():
    try:
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(DUMMY_PDF_PATH)
        c.drawString(100, 750, "This is a test PDF file.")
        c.drawString(100, 700, "It contains some sample text.")
        c.save()
        return True
    except ImportError:
        print("Error: reportlab library is not installed. Cannot create dummy PDF.")
        print("Please install it using: pip install reportlab")
        return False

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))

class TestIngestionAgent(unittest.TestCase):
    def setUp(self):
        self.ingestion_agent = IngestionAgent(embedding_dimension=EMBEDDING_DIMENSION, faiss_index_path=FAISS_INDEX_PATH)
        if not os.path.exists("data/raw"):
            os.makedirs("data/raw")
        if not os.path.exists(DUMMY_PDF_PATH):
            create_dummy_pdf()
            self.assertTrue(os.path.exists(DUMMY_PDF_PATH), "Dummy PDF file not created.")
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH) # Ensure a fresh index for each test

    def tearDown(self):
        if os.path.exists(DUMMY_PDF_PATH):
            os.remove(DUMMY_PDF_PATH)
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)

    def test_extract_text_from_pdf(self):
        if os.path.exists(DUMMY_PDF_PATH):
            extracted_text = self.ingestion_agent.extract_text(DUMMY_PDF_PATH)
            self.assertIn("This is a test PDF file.", extracted_text)
            self.assertIn("It contains some sample text.", extracted_text)
        else:
            self.skipTest("Dummy PDF file not available.")

    def test_chunk_text_by_paragraph(self):
        dummy_text = "This is the first paragraph.\n\nThis is the second paragraph.\n\nAnd this is the third one."
        chunks = self.ingestion_agent.chunk_text(dummy_text)
        self.assertIsInstance(chunks, list)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "This is the first paragraph.")
        self.assertEqual(chunks[1], "This is the second paragraph.")
        self.assertEqual(chunks[2], "And this is the third one.")

    def test_chunk_text_with_token_limit(self):
        long_paragraph = "This is a very long sentence that needs to be split into multiple chunks because it exceeds the token limit. It has many words and continues for quite some time to ensure that the chunking logic based on token count is properly exercised. Let's add even more words and repeat this sentence multiple times to make it definitely longer than our limit. This should result in several smaller chunks. This is a very long sentence that needs to be split into multiple chunks because it exceeds the token limit. It has many words and continues for quite some time to ensure that the chunking logic based on token count is properly exercised. Let's add even more words and repeat this sentence multiple times to make it definitely longer than our limit. This should result in several smaller chunks. We need to add even more text here to really make sure the token count goes well over one hundred."
        chunk_limit = 100
        print(f"Token count of long_paragraph: {count_tokens(long_paragraph)}") # Debug print
        chunks = self.ingestion_agent.chunk_text(long_paragraph, chunk_token_limit=chunk_limit) # Corrected call
        print(f"Number of chunks generated: {len(chunks)}") # Debug print
        print(f"Generated chunks: {chunks}") # Debug print
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 1, "Long paragraph should be split into multiple chunks.")
        for chunk in chunks:
            self.assertLessEqual(count_tokens(chunk), chunk_limit + 10, f"Chunk '{chunk}' exceeds token limit of {chunk_limit}.")

        # Test with shorter sentences within a long paragraph
        mixed_paragraph = "Short sentence. This is a longer sentence that might still fit within the token limit and needs many more words to exceed it significantly. Another short one. And then a very long sentence that will definitely need to be split because it contains an excessive number of words and keeps going on and on and on and on and on and on and on and on and on and on and on and on to ensure proper testing of the splitting mechanism. Let's add even more filler here to be absolutely sure it's much longer than 100 tokens. And let's repeat some of this long sentence again to really push the token count over the limit and ensure proper splitting. And more words here just to be safe."
        print(f"Token count of mixed_paragraph: {count_tokens(mixed_paragraph)}") # Debug print
        chunks_mixed = self.ingestion_agent.chunk_text(mixed_paragraph, chunk_token_limit=chunk_limit) # Corrected call
        print(f"Number of chunks for mixed_paragraph: {len(chunks_mixed)}") # Debug print
        print(f"Generated chunks for mixed_paragraph: {chunks_mixed}") # Debug print
        self.assertIsInstance(chunks_mixed, list)
        self.assertTrue(len(chunks_mixed) > 1, "Mixed paragraph should be split.")
        for chunk in chunks_mixed:
            self.assertLessEqual(count_tokens(chunk), chunk_limit + 10, f"Chunk '{chunk}' in mixed paragraph exceeds limit.")

    def test_chunk_text_with_shorter_limit(self):
        short_paragraph = "This is a short paragraph with a few sentences."
        chunk_limit = 10
        print(f"Token count of short_paragraph: {count_tokens(short_paragraph)}") # Debug print
        chunks = self.ingestion_agent.chunk_text(short_paragraph, chunk_token_limit=chunk_limit)
        print(f"Number of chunks for short_paragraph: {len(chunks)}") # Debug print
        print(f"Generated chunks for short_paragraph: {chunks}") # Debug print
        self.assertIsInstance(chunks, list)
        self.assertGreaterEqual(len(chunks), 1, "Short paragraph should result in at least one chunk.")
        if count_tokens(short_paragraph) > chunk_limit:
            self.assertTrue(len(chunks) > 1, "Short paragraph should be split with a very small limit if it exceeds it.")
            for chunk in chunks:
                self.assertLessEqual(count_tokens(chunk), chunk_limit + 5, f"Short paragraph chunk exceeds limit of {chunk_limit}.")
        else:
            self.assertEqual(len(chunks), 1, "Short paragraph should not be split if it doesn't exceed the limit.")

        single_sentence_long = "This is a single very long sentence that should be split and needs many many words to go over fifteen tokens so that the splitting logic is properly tested."
        print(f"Token count of single_sentence_long: {count_tokens(single_sentence_long)}") # Debug print
        chunks_single = self.ingestion_agent.chunk_text(single_sentence_long, chunk_token_limit=15)
        print(f"Number of chunks for single_sentence_long: {len(chunks_single)}") # Debug print
        print(f"Generated chunks for single_sentence_long: {chunks_single}") # Debug print
        self.assertIsInstance(chunks_single, list)
        self.assertTrue(len(chunks_single) > 1, "Single long sentence should be split.")
        for chunk in chunks_single:
            self.assertLessEqual(count_tokens(chunk), 16, "Single long sentence chunk exceeds limit.")

    def test_embed_chunks(self):
        dummy_chunks = [
            "This is the first test chunk.",
            "Here is the second chunk with more words.",
            "A third, shorter chunk."
        ]
        embeddings = self.ingestion_agent.embed_chunks(dummy_chunks)
        self.assertIsInstance(embeddings, np.ndarray) # Expecting NumPy array now
        self.assertEqual(embeddings.shape[0], len(dummy_chunks))
        self.assertEqual(embeddings.shape[1], EMBEDDING_DIMENSION)

    def test_store_embeddings_in_faiss(self):
        if os.path.exists(DUMMY_PDF_PATH):
            extracted_text = self.ingestion_agent.extract_text(DUMMY_PDF_PATH)
            chunks = self.ingestion_agent.chunk_text(extracted_text, chunk_token_limit=50)
            embeddings = self.ingestion_agent.embed_chunks(chunks)
            self.ingestion_agent.store_embeddings_and_chunks(chunks, embeddings) # Corrected method name

            # Check if the index file exists
            self.assertTrue(os.path.exists(FAISS_INDEX_PATH), "FAISS index file should be created.")

            # Load the index and check the number of vectors
            loaded_index = faiss.read_index(FAISS_INDEX_PATH)
            self.assertEqual(loaded_index.ntotal, len(embeddings), "Number of vectors in FAISS index does not match the number of embeddings.")
        else:
            self.skipTest("Dummy PDF file not available.")

    def test_retrieve_relevant_chunks(self):
        if os.path.exists(DUMMY_PDF_PATH):
            extracted_text = self.ingestion_agent.extract_text(DUMMY_PDF_PATH)
            chunks = self.ingestion_agent.chunk_text(extracted_text, chunk_token_limit=50)
            embeddings = self.ingestion_agent.embed_chunks(chunks)
            self.ingestion_agent.store_embeddings_and_chunks(chunks, embeddings) # Corrected method name

            query = "sample text"
            relevant_chunks = self.ingestion_agent.retrieve_relevant_chunks(query, k=1)
            self.assertIsInstance(relevant_chunks, list)
            self.assertEqual(len(relevant_chunks), 1)
            self.assertIn("sample text", relevant_chunks[0].lower()) # Basic check for content
        else:
            self.skipTest("Dummy PDF file not available.")

if __name__ == '__main__':
    unittest.main()