from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def template_reply(intent, evidence):
    response = evidence[0]["response_text"] if evidence else ""

    if response:
        return (
            "Thanks for reaching out. Based on how similar cases were handled, "
            f"the relevant support path is: {response}"
        )

    return (
        "Thanks for reaching out. "
        "We need a little more information to help with this request."
    )


def llm_reply(customer_text, intent, evidence):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return template_reply(intent, evidence)

    from openai import OpenAI

    # Gemini's OpenAI-compatible API
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    evidence_text = "\n\n".join(
        f"Example {i + 1} customer: {e['customer_text']}\n"
        f"Example {i + 1} support resolution: {e['response_text']}"
        for i, e in enumerate(evidence)
    )

    prompt = f"""
You are a customer-support drafting assistant.

Brand: Apple Support

Predicted intent:
{intent}

Customer message:
{customer_text}

Historical support evidence:
{evidence_text}

Write a short, helpful customer-support reply.

Rules:
- Use only facts and actions supported by the historical evidence.
- Do not invent policies, prices, timelines, URLs, account details, or technical claims.
- Do not claim that you performed an action that you cannot actually perform.
- Do not mention the retrieval system, AI, model, or evaluation process.
- Keep the reply concise and natural.
- If the historical evidence is insufficient to safely answer the customer,
  say that a human support agent should review the request.
"""

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    try:
        out = client.chat.completions.create(
            model=model_name,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You write concise, grounded customer-support replies "
                        "based only on provided historical evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return out.choices[0].message.content.strip()
    except Exception as error:
        # Graceful fallback to grounded historical template reply when API is unavailable or rate-limited
        return template_reply(intent, evidence)