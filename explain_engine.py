def generate_explanation(state, rule_risk, ai_risk):
    reasons = []

    if state["speed"] > 120:
        reasons.append("high speed")

    if state["engine_temp"] > 95:
        reasons.append("elevated engine temperature")

    if abs(state["acceleration"]) > 5:
        reasons.append("aggressive acceleration")

    if not reasons:
        return "Driving conditions are stable."

    explanation = "Risk is elevated due to " + ", ".join(reasons) + "."
    explanation += f" Rule model: {rule_risk}%. AI model: {ai_risk}%."

    return explanation
