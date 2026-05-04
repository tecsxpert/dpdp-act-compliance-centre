import os
import time
import json
import logging
import requests
from dotenv import load_dotenv

load_dotenv()  # MUST be before any os.getenv()

# ── Logging: prints to terminal + saves to groq.log ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("groq.log")
    ]
)
logger = logging.getLogger(__name__)


class GroqClient:

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL        = "llama-3.3-70b-versatile"
    TIMEOUT_SEC  = 30
    MAX_RETRIES  = 3

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY missing. Add it to .env:  GROQ_API_KEY=gsk_xxxx"
            )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json"
        }
        logger.info(f"GroqClient ready — model: {self.MODEL}")

    # -------------------------------------------------------------------------
    # call() — returns plain text string from AI, or None on total failure
    # -------------------------------------------------------------------------
    def call(self, messages: list, temperature: float = 0.3, max_tokens: int = 1000):
        payload = {
            "model":       self.MODEL,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                logger.info(f"Groq call — attempt {attempt + 1}/{self.MAX_RETRIES}")

                response = requests.post(
                    self.GROQ_API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=self.TIMEOUT_SEC
                )
                response.raise_for_status()

                data        = response.json()
                reply_text  = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)

                logger.info(f"Groq success — tokens: {tokens_used}")
                return reply_text

            except requests.exceptions.Timeout:
                logger.warning(f"Attempt {attempt+1}: Timeout after {self.TIMEOUT_SEC}s")

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else "unknown"
                if status == 401:
                    logger.error("Invalid GROQ_API_KEY (401). Fix your .env.")
                    return None   # no point retrying a bad key
                elif status == 429:
                    logger.warning(f"Attempt {attempt+1}: Rate limited (429)")
                else:
                    logger.warning(f"Attempt {attempt+1}: HTTP {status} — {e}")

            except requests.exceptions.ConnectionError:
                logger.warning(f"Attempt {attempt+1}: Network unreachable")

            except (KeyError, IndexError) as e:
                logger.error(f"Attempt {attempt+1}: Bad response structure — {e}")

            except Exception as e:
                logger.error(f"Attempt {attempt+1}: Unexpected error — {e}")

            # Exponential backoff: 1s → 2s → 4s
            if attempt < self.MAX_RETRIES - 1:
                wait = 2 ** attempt
                logger.info(f"Waiting {wait}s before retry...")
                time.sleep(wait)

        logger.error("All 3 Groq attempts failed. Returning None.")
        return None

    # -------------------------------------------------------------------------
    # call_for_json() — returns parsed dict, strips ```json fences automatically
    # Use this in /categorise, /generate-report endpoints
    # -------------------------------------------------------------------------
    def call_for_json(self, messages: list, temperature: float = 0.1, max_tokens: int = 1000):
        raw = self.call(messages, temperature=temperature, max_tokens=max_tokens)
        if raw is None:
            return None
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
                cleaned = cleaned.rsplit("```", 1)[0].strip()
            parsed = json.loads(cleaned)
            logger.info("JSON parsed from Groq response successfully")
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e} | Raw: {raw[:200]}")
            return None
