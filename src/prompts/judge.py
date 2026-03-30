from langchain_core.prompts import ChatPromptTemplate

rev_prompt = ChatPromptTemplate.from_template(
"""
You are a careful evaluator.

Task:
Given a DOCUMENT and a CLAIM, determine how strongly the DOCUMENT supports the CLAIM.

Return ONLY valid JSON in this exact format:
{{"score": <float between 0 and 1>, "label": <0 or 1>}}

Scoring guide:
- 1.0 = fully supported by the document
- 0.8 = mostly supported
- 0.5 = partially supported / uncertain
- 0.2 = weakly supported
- 0.0 = not supported or contradicted

Rules:
- Use only the document.
- Do not use outside knowledge.
- If the claim is not clearly supported, give a low score.
- Output JSON only.

DOCUMENT:
{doc}

CLAIM:
{claim}
"""
)