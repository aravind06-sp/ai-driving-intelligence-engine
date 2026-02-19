import random

class VehicleEngine:
    def __init__(self):
        self.speed = 0
        self.fuel = 100
        self.engine_temp = 70
        self.rpm = 800
        self.acceleration = 0

    def update(self):
        throttle = random.uniform(-5, 8)
        self.acceleration = throttle

        self.speed += self.acceleration
        self.speed = max(0, min(self.speed, 180))

        self.rpm = 800 + self.speed * 40

        fuel_drop = 0.05 * self.speed / 40
        self.fuel -= fuel_drop
        self.fuel = max(self.fuel, 0)

        temp_change = 0.1 * self.rpm / 1000
        self.engine_temp += temp_change

        if self.speed < 20:
            self.engine_temp -= 0.5

        self.engine_temp = max(60, min(self.engine_temp, 130))

    def get_state(self):
        return {
            "speed": round(self.speed, 2),
            "fuel": round(self.fuel, 2),
            "engine_temp": round(self.engine_temp, 2),
            "rpm": round(self.rpm, 2),
            "acceleration": round(self.acceleration, 2)
        }
