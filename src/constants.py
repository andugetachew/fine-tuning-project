"""
Shared constants used by both data prep and the inference API, so the
system prompt used at training time matches what's used at serving time.
"""

SYSTEM_PROMPT = (
    "You are a precise, practical backend engineering assistant "
    "specializing in Python, Django, and REST API design."
)
