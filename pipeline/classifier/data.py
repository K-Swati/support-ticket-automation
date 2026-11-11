"""Synthetic support-ticket generator.

There is no public dataset behind this project, so tickets are generated from
phrase templates with random slot values and realistic noise (typos, dropped
words, filler). The **last two templates of every intent are held out** and
only used to build the evaluation set, so the reported accuracy measures
generalisation to unseen phrasings and not memorisation of training templates.

Because the data is synthetic, the accuracy it produces says nothing about
performance on real customer text. The point of the project is the pipeline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

PLANS = ["Basic", "Pro", "Team", "Business", "Starter"]
AMOUNTS = ["$19", "$49", "$99", "$149", "$29.99"]
FEATURES = ["dashboard", "export", "mobile app", "search", "reports", "notifications", "API"]
DEVICES = ["laptop", "phone", "tablet", "work computer"]

TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "I was charged twice for my {plan} plan this month",
        "why is there an extra {amount} charge on my invoice",
        "please send me a copy of last month's invoice",
        "I need a refund for the {amount} payment taken yesterday",
        "my credit card was billed the wrong amount",
        "can you update the card on file for my subscription payments",
        "the price on my bill is higher than the {plan} plan price",
        "I never received a receipt after paying {amount}",
        "there is a duplicate transaction on my statement",
        "what taxes are included in my monthly invoice",
    ],
    "technical_issue": [
        "the {feature} keeps crashing when I open it",
        "I get an error message every time I try to use {feature}",
        "the app is very slow on my {device} since the update",
        "{feature} stopped working and shows a blank page",
        "uploading files fails with a timeout error",
        "the page won't load on my {device} at all",
        "I found a bug where {feature} loses my changes",
        "sync between my {device} and the web app is broken",
        "notifications are not arriving and the {feature} looks frozen",
        "the website returns a 500 error when I save",
    ],
    "account_access": [
        "I forgot my password and can't log in",
        "my account is locked after too many sign in attempts",
        "I'm not getting the verification code to log in",
        "how do I change the email address on my account",
        "I can't sign in on my new {device}",
        "two factor authentication is not letting me in",
        "please reset my login, I lost access to my account",
        "my username is not recognised when I try to sign in",
        "I need to add a teammate to my account with login access",
        "the password reset link has expired",
    ],
    "cancellation": [
        "I want to cancel my {plan} subscription",
        "please close my account and stop all billing",
        "how do I end my membership before it renews",
        "I'd like to terminate my plan at the end of this month",
        "cancel my subscription effective immediately",
        "I no longer need the service so please cancel it",
        "stop my renewal, I am leaving the {plan} plan",
        "how can I unsubscribe and delete everything",
        "I'm switching providers and need to cancel my service",
        "end my {plan} plan and confirm the cancellation in writing",
    ],
    "general_inquiry": [
        "do you offer a discount for non profit organizations",
        "what are your support hours",
        "does the {plan} plan include the {feature}",
        "can you tell me more about your enterprise options",
        "is there a student pricing option",
        "where can I find documentation about the {feature}",
        "which countries do you operate in",
        "do you have a partner program",
        "how many users are included in the {plan} plan",
        "is there a free trial available",
    ],
}
HELD_OUT = 2  # templates per intent reserved for evaluation only

FILLERS_PREFIX = ["hi,", "hello,", "hey team,", "good morning,", "", "", ""]
FILLERS_SUFFIX = ["thanks", "thank you", "please help", "asap", "", "", ""]


@dataclass(frozen=True)
class Example:
    text: str
    intent: str


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        plan=rng.choice(PLANS), amount=rng.choice(AMOUNTS), feature=rng.choice(FEATURES), device=rng.choice(DEVICES)
    )


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def add_noise(text: str, rng: random.Random, typo_rate: float = 0.06, drop_rate: float = 0.04) -> str:
    out = []
    for word in text.split():
        r = rng.random()
        if r < drop_rate and len(word) > 3:
            continue
        out.append(_typo(word, rng) if r < drop_rate + typo_rate else word)
    sentence = " ".join(out)
    sentence = f"{rng.choice(FILLERS_PREFIX)} {sentence} {rng.choice(FILLERS_SUFFIX)}".strip()
    return sentence.lower() if rng.random() < 0.3 else sentence


def generate(n_per_intent: int, seed: int, *, held_out: bool = False, noise: bool = True) -> list[Example]:
    """Generate ``n_per_intent`` tickets per intent from training templates, or held-out templates."""
    rng = random.Random(seed)
    examples: list[Example] = []
    for intent, templates in TEMPLATES.items():
        pool = templates[-HELD_OUT:] if held_out else templates[:-HELD_OUT]
        for _ in range(n_per_intent):
            text = _fill(rng.choice(pool), rng)
            examples.append(Example(add_noise(text, rng) if noise else text, intent))
    rng.shuffle(examples)
    return examples
