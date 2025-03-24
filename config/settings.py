import os

# MongoDB settings
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "gov_website")

# Crawler settings
DEFAULT_CRAWL_INTERVAL = 1.0
DEFAULT_MAX_PAGES = None
DEFAULT_MAX_DEPTH = None
DEFAULT_INCLUDE_SUBDOMAINS = True

# LLM settings
USE_LLM = os.getenv("USE_LLM", "False").lower() == "true"
# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
# Custom model settings
CUSTOM_MODEL_URL = os.getenv("CUSTOM_MODEL_URL", "")
CUSTOM_MODEL_API_KEY = os.getenv("CUSTOM_MODEL_API_KEY", "")

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))