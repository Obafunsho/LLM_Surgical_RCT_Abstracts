import os  # Import standard library to work with environment variables and file paths

# OpenAI API key is loaded from an environment variable (recommended for security)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Raise an error if the key is not found to avoid silent failure
if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# URL for PubMed Central API to fetch abstracts
PMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Directory containing original (unmodified) abstracts
INPUT_DIR_ORIGINAL = "data/original"

# Directory containing rewritten (LLM-enhanced) abstracts
INPUT_DIR_REWRITTEN = "data/rewritten"

# Directory to store scoring results for original abstracts
OUTPUT_DIR_ORIGINAL = "data/scores/original"

# Directory to store scoring results for rewritten abstracts
OUTPUT_DIR_REWRITTEN = "data/scores/rewritten"