"""Explicit adapter doubles for composition tests that do not judge semantics."""


def supported_semantics(prompt):
    """Assume support for existing orchestration regressions only.

    Runtime semantic rejection tests inject their own verdicts and exercise the
    real validator. This helper is never imported by production code.
    """
    return {"support": "supported", "explicit_absence": "clear"}


def complete_coverage(prompt):
    """Assume coverage only in existing tests focused on other behavior."""
    import json

    payload = json.loads(prompt.split("INPUT DATA:\n", 1)[1])
    context = payload["research_context"]
    status = "conflict" if context["conflicts"] else (
        "gap" if context["unresolved_claims"] else "answered")
    assessments = []
    limitations = context.get("validation_limitations", [])
    for i, requirement in enumerate(payload["required_claims"] or [payload["question"]]):
        indices = [j for j, item in enumerate(limitations)
                   if not payload["required_claims"] or i in item["requirement_indices"]]
        if indices:
            assessments.append({"requirement": requirement, "status": "validation_limited",
                "answer_excerpt": limitations[indices[0]]["message"], "limitation_indices": indices})
        else:
            assessments.append({"requirement": requirement, "status": status,
                "answer_excerpt": payload["answer"],
                "citation_claim_indices": list(range(len(payload["citation_claims"]))) if status == "answered" else [],
                "gap_indices": [0] if status == "gap" else [],
                "conflict_indices": [0] if status == "conflict" else []})
    return {"coverage": "complete", "presentation": assessments}
