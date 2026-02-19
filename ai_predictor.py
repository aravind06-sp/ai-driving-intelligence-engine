import torch
import torch.nn as nn

device = torch.device("cuda")

class RiskPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

model = RiskPredictor().to(device)
model.load_state_dict(torch.load("risk_model.pth"))
model.eval()

def predict_risk(speed, temp, accel):
    x = torch.tensor([[speed, temp, accel]], dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(x)
    return round(float(prediction.item()), 2)
