"""
Answer generation via the Gemini API.

The system prompt lives in system_instruction (not the user turn) so the
model treats grounding rules as a hard constraint rather than a suggestion.
Uses the google-genai unified SDK (pip install google-genai).
"""

import logging

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

# Module-level client, reused across all requests (thread-safe).
_client = genai.Client(api_key=GEMINI_API_KEY)

_SYSTEM_PROMPT = """You are a precise document assistant. Your only job is to answer
questions using the context passages provided below. Rules you must follow:

1. Base every statement on the provided context — never on prior knowledge.
2. Cite the page number and filename for every claim (e.g. "Page 4, report.pdf").
3. If the answer is not present in the context, respond with exactly:
   "This information is not found in the uploaded document."
4. Be concise. Avoid filler phrases like "Based on the context provided...".
"""


def generate_answer(query: str, context: list[dict]) -> str:
    """
    Build a prompt from retrieved chunks and call Gemini.

    Each chunk is labelled [Page N | filename] so the model can cite them.
    """
    if not context:
        return "No relevant content was found in the document for this question."

    context_block = "\n\n".join(
        f"[Page {c['page']} | {c['source']}]\n{c['content']}"
        for c in context
    )

    prompt = f"""--- CONTEXT START ---
{context_block}
--- CONTEXT END ---

Question: {query}

Answer:"""

    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=1024,
                temperature=0.2,  # low temperature keeps answers faithful to context
            ),
        )
        return response.text

    except Exception as e:
        logger.exception("Gemini API call failed")
        raise RuntimeError(f"Answer generation failed: {e}") from e
