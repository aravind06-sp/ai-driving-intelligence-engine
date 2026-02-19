import time
from vehicle_engine import VehicleEngine
from risk_engine import RiskEngine
from ai_predictor import predict_risk
from explain_engine import generate_explanation

car = VehicleEngine()
risk_model = RiskEngine()

while True:
    car.update()
    state = car.get_state()

    rule_analysis = risk_model.evaluate(state)
    ai_risk = predict_risk(
        state["speed"],
        state["engine_temp"],
        state["acceleration"]
    )

    explanation = generate_explanation(
        state,
        rule_analysis["risk_score"],
        ai_risk
    )

    print(
        "Speed:", state["speed"],
        "| Temp:", state["engine_temp"],
        "| Rule Risk:", rule_analysis["risk_score"],
        "| AI Risk:", ai_risk,
        "| Status:", rule_analysis["status"]
    )

    print("Explanation:", explanation)
    print("-" * 60)

    time.sleep(1)
