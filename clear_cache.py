import os
import shutil
from pathlib import Path

def clear_hf_cache():
    """Clears the Hugging Face cache directory."""
    print("Inside clear_hf_cache() function")  # ADD THIS LINE
    cache_dir = Path.home() / ".cache" / "huggingface"
    print(f"Hugging Face cache directory: {cache_dir}")

    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print("Hugging Face cache cleared.")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    else:
        print("Hugging Face cache directory not found.")

# Call the function to clear the cache (uncomment to run)
clear_hf_cache()