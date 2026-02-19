class RiskEngine:
    def __init__(self):
        self.history = []

    def evaluate(self, state):
        speed = state["speed"]
        temp = state["engine_temp"]
        accel = abs(state["acceleration"])

        speed_risk = min(speed / 180 * 50, 50)
        temp_risk = max((temp - 85) * 3, 0)
        accel_risk = min(accel * 6, 30)

        total_risk = speed_risk + temp_risk + accel_risk
        total_risk = max(0, min(total_risk, 100))

        efficiency = max(0, 100 - total_risk)

        status = "Normal"
        if total_risk > 70:
            status = "High Risk"
        elif total_risk > 40:
            status = "Moderate Risk"

        return {
            "risk_score": round(total_risk, 2),
            "efficiency": round(efficiency, 2),
            "status": status
        }
