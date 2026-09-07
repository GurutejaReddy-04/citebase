"""
Grounded answer generation via the Gemini API.

Synthesizes context-grounded responses with strict bracketed numeric citations.
"""

import logging
import time

import threading
from typing import Optional

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

_client: Optional[genai.Client] = None
_client_lock = threading.Lock()


def _get_client() -> genai.Client:
    """Lazy-loaded thread-safe singleton for the Gemini client."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client

_SYSTEM_PROMPT = """You are CiteBase, an enterprise document intelligence engine. Your only job is to answer
questions using the numbered context passages provided below. Rules you must follow:

1. Base every statement strictly on the provided context — never on prior knowledge.
2. Cite sources using numeric bracket references like [1], [2] directly in the text where each claim is made.
3. NEVER put URLs, website links, or full document titles in the answer text itself — use ONLY the numeric bracket references (e.g. [1], [2]) matching the numbered sources below.
4. When answering questions spanning multiple sources, synthesize findings and clearly attribute each fact to its corresponding numeric source [1], [2].
5. If the answer is not present in the context, respond with exactly:
   "This information is not found in the provided sources."
6. Be concise and factual. Avoid filler phrases like "Based on the context provided...".
"""


def generate_answer(query: str, context: list[dict]) -> str:
    """
    Build a prompt from retrieved chunks (documents or web) and call Gemini.

    Each chunk is numbered [1], [2], etc. so the model can cite cleanly with bracket numbers.
    """
    if not context:
        return "No relevant content was found in the sources for this question."

    context_lines = []
    for idx, c in enumerate(context, 1):
        if c.get("source_type") == "web":
            header = f"[{idx}] (Web Source: {c.get('title') or c.get('doc_title') or 'Web'} | URL: {c.get('url', c.get('source', ''))})"
        else:
            header = f"[{idx}] (Document: {c.get('doc_title') or c.get('source')} | Page: {c.get('page', 1)} | Section: {c.get('section', 'General')})"
        context_lines.append(f"{header}\n{c['content']}")

    context_block = "\n\n".join(context_lines)

    prompt = f"""--- CONTEXT START ---
{context_block}
--- CONTEXT END ---

Question: {query}

Answer:"""

    max_retries = 3
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            client = _get_client()
            # Placed in system_instruction because models treat user-turn constraints as polite suggestions.
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    max_output_tokens=1024,
                    temperature=0.2,
                    http_options=types.HttpOptions(timeout=30000),
                ),
            )
            return response.text or ""

        except Exception as e:
            last_error = e
            err_msg = str(e)
            if ("503" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < max_retries:
                backoff = attempt * 1.5
                logger.warning("Gemini API transient error (%s). Retrying in %.1fs (attempt %d/%d)...", err_msg[:80], backoff, attempt, max_retries)
                time.sleep(backoff)
                continue

            logger.exception("Gemini API generation call failed: %s", e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                logger.warning("Gemini quota exhausted. Returning grounded context fallback.")
                if context:
                    top_chunk = context[0].get("content", "")
                    return f"Grounded response: {top_chunk[:300]} [1]"
                return "This information is not found in the provided sources."
            raise RuntimeError(f"Gemini LLM generation failed: {e}") from e

    raise RuntimeError(f"Gemini LLM generation failed after {max_retries} attempts: {last_error}") from last_error

