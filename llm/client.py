"""LLM Client for Google Gemini API with retry logic."""

import time
from google import genai

from config import GEMINI_API_KEY, DEFAULT_MODEL, MAX_RETRIES, RETRY_DELAY
from utils import get_logger

logger = get_logger(__name__)

import os
from dotenv import load_dotenv

load_dotenv()

# Load your API key from .env
# GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")


class LLMClient:

    def __init__(self, model_name=DEFAULT_MODEL):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text

    def call(self, prompt: str) -> str:
        return self.generate(prompt)