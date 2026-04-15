"""
Strict query scope guard for the medical insurance assistant.

This guard allows a query only when it is:
1. Clearly within the medical/health insurance domain, or
2. A follow-up that is still about the current in-scope conversation topic.
"""

from typing import Optional

from openai import OpenAI


FALLBACK_MESSAGE = (
    "The topic is not related to medical insurance. Please ask an insurance-related question."
)


class QueryScopeGuard:
    """Determines whether a query is in scope before answer generation."""

    DECISION_IN_SCOPE = "in_scope"
    DECISION_OUT_OF_SCOPE = "out_of_scope"

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.system_prompt = """You are a strict scope checker for a medical insurance assistant.

Your job is to decide whether the latest user query is allowed.

ALLOW the query only if at least one of these is true:
- The query is clearly about medical insurance, health insurance, mediclaim, coverage, exclusions, claims, premiums, waiting periods, hospitals, reimbursements, cashless treatment, portability, deductibles, co-pay, policy wording, insurer procedures, or similar insurance topics.
- The query is a clear follow-up to the current conversation topic, and that current topic is already medical insurance related.

REJECT the query if it is unrelated, including:
- greetings, small talk, jokes, or casual chat
- sports, news, politics, entertainment, shopping, travel, coding, math, general knowledge, or unrelated productivity requests
- medical advice that is not about insurance coverage or insurance processes
- vague prompts that are not clearly tied to medical insurance or the current insurance-topic context

Important rules:
- Be strict. When uncertain, reject.
- A follow-up is allowed only if it is plausibly connected to the provided current topic context.
- The current topic context is only supporting evidence. Do not allow unrelated topic-switches just because prior context exists.

Respond with exactly one token:
- in_scope
- out_of_scope"""

    def is_in_scope(
        self,
        query: str,
        current_topic_query: Optional[str] = None,
        current_topic_answer: Optional[str] = None,
    ) -> bool:
        topic_query = (current_topic_query or "").strip() or "None"
        topic_answer = (current_topic_answer or "").strip() or "None"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Current in-scope topic query: {topic_query}\n"
                    f"Current in-scope topic answer: {topic_answer}\n"
                    f"Latest user query: {query}"
                ),
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=8,
            )
            if response and response.choices:
                decision = (response.choices[0].message.content or "").strip().lower()
                return decision == self.DECISION_IN_SCOPE
        except Exception as exc:
            print(f"[ScopeGuard] Error during scope check: {exc}")

        # Fail closed so out-of-domain queries are not answered on transient guard failures.
        return False
