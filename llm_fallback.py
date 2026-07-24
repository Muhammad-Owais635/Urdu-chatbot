"""
Optional LLM-powered fallback for messages that don't match a known intent
with sufficient confidence. Disabled by default (LLM_FALLBACK_ENABLED=false)
since it requires an API key and incurs cost — the bot works fully without it,
using the "unknown" response template instead.

Uses OpenAI's API by default; swap the client call if you prefer another
provider (Anthropic, a local model via Ollama, etc.) — the interface
(generate_fallback_response) is provider-agnostic.
"""

import logging

logger = logging.getLogger(__name__)


class LLMFallback:
    def __init__(self, config):
        self.enabled = config.LLM_FALLBACK_ENABLED
        self.api_key = config.OPENAI_API_KEY
        self.model = config.LLM_MODEL
        self.system_prompt = config.LLM_SYSTEM_PROMPT
        self._client = None

        if self.enabled and not self.api_key:
            logger.warning(
                "LLM_FALLBACK_ENABLED is true but OPENAI_API_KEY is missing. "
                "Falling back to template responses instead."
            )
            self.enabled = False

        if self.enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning(
                    "openai package not installed. Run "
                    "`pip install openai` to enable LLM fallback. "
                    "Falling back to template responses instead."
                )
                self.enabled = False

    def generate_fallback_response(self, message: str, history: list) -> str | None:
        """
        Returns an LLM-generated response for a message that didn't match
        any known intent, or None if the fallback is disabled/unavailable
        (caller should use the static "unknown" template in that case).
        """
        if not self.enabled or not self._client:
            return None

        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.6,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001 — log and gracefully degrade
            logger.error("LLM fallback call failed: %s", exc)
            return None
