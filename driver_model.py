class DriverModel:
    def __init__(self):
        self.total_accel = 0
        self.total_speed = 0
        self.samples = 0

    def update(self, state):
        self.total_accel += abs(state["acceleration"])
        self.total_speed += state["speed"]
        self.samples += 1

    def get_profile(self):
        if self.samples == 0:
            return "Unknown"

        avg_accel = self.total_accel / self.samples
        avg_speed = self.total_speed / self.samples

        score = avg_accel * 5 + avg_speed / 5

        if score > 80:
            return "Aggressive Driver"
        elif score > 40:
            return "Moderate Driver"
        else:
            return "Calm Driver"
