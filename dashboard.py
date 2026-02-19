import tkinter as tk
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from vehicle_engine import VehicleEngine
from risk_engine import RiskEngine
from ai_predictor import predict_risk
from explain_engine import generate_explanation
from driver_model import DriverModel

car = VehicleEngine()
risk_model = RiskEngine()
driver_model = DriverModel()

root = tk.Tk()
root.title("AI Driving Intelligence Engine")
root.geometry("900x650")
root.configure(bg="#121212")

def create_label(size=14, bold=False):
    return tk.Label(
        root,
        font=("Arial", size, "bold" if bold else "normal"),
        bg="#121212",
        fg="white"
    )

speed_label = create_label()
temp_label = create_label()
rule_label = create_label()
ai_label = create_label()
driver_label = create_label(16, True)
status_label = create_label(16, True)
explain_label = tk.Label(
    root,
    wraplength=850,
    font=("Arial", 11),
    bg="#121212",
    fg="lightgray"
)

speed_label.pack()
temp_label.pack()
rule_label.pack()
ai_label.pack()
driver_label.pack(pady=5)
status_label.pack(pady=5)
explain_label.pack(pady=5)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(7, 3))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

rule_history = deque(maxlen=60)
ai_history = deque(maxlen=60)

def update_dashboard():
    car.update()
    state = car.get_state()

    driver_model.update(state)

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

    rule_history.append(rule_analysis["risk_score"])
    ai_history.append(ai_risk)

    speed_label.config(text=f"Speed: {state['speed']} km/h")
    temp_label.config(text=f"Temperature: {state['engine_temp']} °C")
    rule_label.config(text=f"Rule Risk: {rule_analysis['risk_score']} %")
    ai_label.config(text=f"AI Risk: {ai_risk} %")
    driver_label.config(text=f"Driver Profile: {driver_model.get_profile()}")
    status_label.config(text=f"Status: {rule_analysis['status']}")
    explain_label.config(text=f"Explanation: {explanation}")

    if rule_analysis["risk_score"] > 70:
        status_label.config(fg="red")
    elif rule_analysis["risk_score"] > 40:
        status_label.config(fg="orange")
    else:
        status_label.config(fg="lightgreen")

    ax.clear()
    ax.plot(rule_history, label="Rule Risk")
    ax.plot(ai_history, label="AI Risk")
    ax.set_ylim(0, 100)
    ax.set_title("Risk Trend Over Time")
    ax.legend()
    canvas.draw()

    root.after(1000, update_dashboard)

update_dashboard()
root.mainloop()
