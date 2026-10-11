import logging
import re
import json
from pathlib import Path
from datetime import datetime
from config import LOGS_DIR

def get_logger(name: str) -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    logfile = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers on repeated imports
    if not logger.handlers:
        fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(logfile, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text string. Returns empty dict if not found."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
    except json.JSONDecodeError:
        return {}

def load_file(file_path: str) -> str:
    """Load and return file contents as string."""
    return Path(file_path).read_text().strip()