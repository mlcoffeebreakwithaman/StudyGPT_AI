import os
import pathlib

def create_comprehensive_project_structure(base_path="StudyGPT_AI"):
    """Creates a comprehensive folder structure for the StudyGPT AI project."""

    folders = [
        os.path.join(base_path, "core", "agents"),
        os.path.join(base_path, "core", "chains"),
        os.path.join(base_path, "core", "llm_utils"),
        os.path.join(base_path, "core", "vector_db"),
        os.path.join(base_path, "core", "data_access"),
        os.path.join(base_path, "core", "utils"),
        os.path.join(base_path, "prompts"),
        os.path.join(base_path, "config"),
        os.path.join(base_path, "data", "raw"),
        os.path.join(base_path, "data", "interim"),
        os.path.join(base_path, "data", "processed"),
        os.path.join(base_path, "vector_index", "faiss"),
        os.path.join(base_path, "evaluation"),
        os.path.join(base_path, "logs"),
        os.path.join(base_path, "tests"),
        os.path.join(base_path, "notebooks"),
        os.path.join(base_path, "api_docs"),
        base_path,
    ]

    init_files = [
        os.path.join(base_path, "core", "agents", "__init__.py"),
        os.path.join(base_path, "core", "chains", "__init__.py"),
        os.path.join(base_path, "core", "llm_utils", "__init__.py"),
        os.path.join(base_path, "core", "vector_db", "__init__.py"),
        os.path.join(base_path, "core", "data_access", "__init__.py"),
        os.path.join(base_path, "core", "utils", "__init__.py"),
        os.path.join(base_path, "prompts", "__init__.py"),
        os.path.join(base_path, "config", "__init__.py"),
        os.path.join(base_path, "data", "__init__.py"),
        os.path.join(base_path, "logs", "__init__.py"),
        os.path.join(base_path, "tests", "__init__.py"),
        os.path.join(base_path, "notebooks", "__init__.py"),
        os.path.join(base_path, "api_docs", "__init__.py"),
        os.path.join(base_path, "evaluation", "__init__.py"),
        os.path.join(base_path, "vector_index", "__init__.py"),
        os.path.join(base_path, "vector_index", "faiss", "__init__.py"),
        os.path.join(base_path, "core", "__init__.py"),
    ]

    files = [
        os.path.join(base_path, "core", "agents", "ingestion_agent.py"),
        os.path.join(base_path, "core", "agents", "textbook_agent.py"),
        os.path.join(base_path, "core", "agents", "tutor_agent.py"),
        os.path.join(base_path, "core", "agents", "quiz_agent.py"),
        os.path.join(base_path, "core", "agents", "progress_agent.py"),
        os.path.join(base_path, "core", "agents", "recommendation_agent.py"),
        os.path.join(base_path, "core", "agents", "memory_agent.py"),
        os.path.join(base_path, "core", "agents", "content_filter_agent.py"),
        os.path.join(base_path, "core", "agents", "reward_agent.py"),
        os.path.join(base_path, "core", "agents", "scheduler_agent.py"),
        os.path.join(base_path, "core", "agents", "monitoring_agent.py"),
        os.path.join(base_path, "core", "chains", "tutoring_chain.py"),
        os.path.join(base_path, "core", "chains", "quiz_generation_chain.py"),
        os.path.join(base_path, "core", "chains", "multi_agent_orchestrator.py"),
        os.path.join(base_path, "core", "llm_utils", "llm_wrapper.py"),
        os.path.join(base_path, "core", "vector_db", "faiss_utils.py"),
        os.path.join(base_path, "core", "data_access", "database_utils.py"),
        os.path.join(base_path, "core", "utils", "error_handling.py"),
        os.path.join(base_path, "core", "utils", "logging_utils.py"),
        os.path.join(base_path, "prompts", "tutor_prompt.txt"),
        os.path.join(base_path, "prompts", "quiz_prompt.txt"),
        os.path.join(base_path, "prompts", "recommendation_prompt.txt"),
        os.path.join(base_path, "config", "agent_config.yaml"),
        os.path.join(base_path, "config", "kb_config.yaml"),
        os.path.join(base_path, "config", "database_config.yaml"),
        os.path.join(base_path, "config", "ingestion_config.yaml"),
        os.path.join(base_path, "config", "llm_config.yaml"),
        os.path.join(base_path, "config", "retrieval_config.yaml"),
        os.path.join(base_path, "evaluation", "evaluation_utils.py"),
        os.path.join(base_path, "evaluation", "metrics.py"),
        os.path.join(base_path, "logs", ".gitkeep"),
        os.path.join(base_path, "tests", "test_ingestion.py"),
        os.path.join(base_path, "tests", "test_retrieval.py"),
        os.path.join(base_path, "tests", "test_tutor_agent.py"),
        os.path.join(base_path, "notebooks", ".gitkeep"),
        os.path.join(base_path, "api_docs", "openapi.yaml"),
        os.path.join(base_path, "api_docs", "endpoint_contracts.md"),
        os.path.join(base_path, "api_docs", "usage_examples.http"),
        os.path.join(base_path, "data", ".gitkeep"),
        os.path.join(base_path, "data", "raw", ".gitkeep"),
        os.path.join(base_path, "data", "interim", ".gitkeep"),
        os.path.join(base_path, "data", "processed", ".gitkeep"),
        os.path.join(base_path, "vector_index", ".gitkeep"),
        os.path.join(base_path, "vector_index", "faiss", ".gitkeep"),
        os.path.join(base_path, "requirements.txt"),
        os.path.join(base_path, ".env"),
        os.path.join(base_path, "README.md"),
        os.path.join(base_path, ".gitignore"),
        os.path.join(base_path, "setup.py"),
        os.path.join(base_path, "pyproject.toml"),
        os.path.join(base_path, "run_ingestion.py"),
        os.path.join(base_path, "run_retrieval_test.py"),
        os.path.join(base_path, "main_ai.py"),
    ]

    root_files_content = {
        "README.md": "# StudyGPT AI Project\n\nComprehensive project structure.",
        ".gitignore": ".venv/\n__pycache__/\n*.pyc\n.env\nlogs/\ndata/interim/\ndata/processed/\nvector_index/\n",
        "requirements.txt": "langchain\nsentence-transformers\nfaiss-cpu\npdfplumber\nPymuPDF\nPyYAML\nrequests\ntqdm\ntransformers\npython-dotenv",
        "setup.py": """from setuptools import setup, find_packages

setup(
    name='studygpt_ai',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'langchain',
        'sentence-transformers',
        'faiss-cpu',
        'pdfplumber',
        'PymuPDF',
        'PyYAML',
        'requests',
        'tqdm',
        'transformers',
        'python-dotenv',
    ],
)
""",
        "pyproject.toml": """[tool.poetry]
name = "studygpt_ai"
version = "0.1.0"
description = "StudyGPT AI Project"
authors = ["Your Name <you@example.com>"]
license = "MIT"
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.9"
langchain = "*"
sentence-transformers = "*"
faiss-cpu = "*"
pdfplumber = "*"
PyMuPDF = "*"
PyYAML = "*"
requests = "*"
tqdm = "*"
transformers = "*"
python-dotenv = "*"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""
    }

    os.makedirs(base_path, exist_ok=True)
    print(f"📦 Created base directory: {base_path}")

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 Created directory: {folder}")

    for init_file in init_files:
        pathlib.Path(init_file).touch()
        print(f"📄 Created __init__.py: {init_file}")

    for file in files:
        pathlib.Path(file).touch()
        print(f"📄 Created empty file: {file}")

    for filename, content in root_files_content.items():
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                f.write(content)
            print(f"📄 Created file with content: {filepath}")

    print("\n✅ Project structure created successfully.")
    print("💡 Next steps: activate your virtual environment, then install dependencies.")

if __name__ == "__main__":
    create_comprehensive_project_structure()
