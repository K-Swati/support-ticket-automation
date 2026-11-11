"""Response generation: a deterministic template responder and an optional OpenAI one."""

from __future__ import annotations

from typing import Protocol

TEMPLATES = {
    "billing": "Thanks for reaching out about billing. We're reviewing the charges on your account and will send a "
    "corrected invoice or refund confirmation within 2 business days.",
    "technical_issue": "Sorry you're running into a problem. We've logged it for our engineers. Please reply with the "
    "steps that lead to the error and any screenshots so we can reproduce it.",
    "account_access": "We can help you get back in. Use the 'Forgot password' link on the sign-in page; if that does "
    "not work, reply here and we'll verify your identity and reset access.",
    "cancellation": "We're sorry to see you go. We've noted your cancellation request and will confirm the end date by "
    "email. You keep access until the end of the current billing period.",
    "general_inquiry": "Thanks for your question! Our team will reply shortly. In the meantime, you'll find plans and "
    "documentation on our website.",
}


class Responder(Protocol):
    name: str

    def respond(self, intent: str, text: str) -> str: ...


class TemplateResponder:
    name = "template"

    def respond(self, intent: str, text: str) -> str:
        return TEMPLATES.get(intent, TEMPLATES["general_inquiry"])


class OpenAIResponder:
    """Drafts a reply with an LLM. Falls back to the template if the model returns nothing."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", client=None) -> None:
        self.model = model
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self._fallback = TemplateResponder()

    def respond(self, intent: str, text: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You write short, polite customer-support replies. Never promise refunds or "
                    f"specific dates. The ticket was classified as '{intent}'.",
                },
                {"role": "user", "content": text},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        return content or self._fallback.respond(intent, text)


def get_responder(kind: str, model: str = "gpt-4o-mini") -> Responder:
    if kind == "template":
        return TemplateResponder()
    if kind == "openai":
        return OpenAIResponder(model)
    raise ValueError(f"unknown responder {kind!r}; choose 'template' or 'openai'")
