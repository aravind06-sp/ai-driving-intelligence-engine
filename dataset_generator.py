import random
import csv

def generate_dataset(samples=5000):
    with open("driving_data.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["speed", "temp", "accel", "risk"])

        for _ in range(samples):
            speed = random.uniform(0, 180)
            temp = random.uniform(60, 130)
            accel = random.uniform(-8, 8)

            speed_risk = min(speed / 180 * 50, 50)
            temp_risk = max((temp - 85) * 3, 0)
            accel_risk = min(abs(accel) * 6, 30)

            total_risk = speed_risk + temp_risk + accel_risk
            total_risk = max(0, min(total_risk, 100))

            writer.writerow([speed, temp, accel, total_risk])

if __name__ == "__main__":
    generate_dataset()
