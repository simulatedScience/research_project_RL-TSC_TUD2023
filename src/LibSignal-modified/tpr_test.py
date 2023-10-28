from generator.lane_vehicle import simulate_true_positives, simulate_false_positives

# create plot showing true positive rate vs. number of vehicles detected
import matplotlib.pyplot as plt

def test_true_positives():
    n_vehicles = 1000
    x_values = [0.0, 0.3, 0.6, 0.8, 0.95, 1.0]
    y_values = []
    for tpr in x_values:
        y_values.append(simulate_true_positives(n_vehicles, tpr)/n_vehicles)
    plt.plot(x_values, y_values, ".-", label="True Positive Rate")
    plt.xlabel("True Positive Rate")
    plt.ylabel("Fraction of Vehicles Detected")
    plt.title("True Positive Rate vs. Number of Vehicles Detected")
    plt.legend(loc='best')
    plt.grid(color="#dddddd")
    plt.tight_layout()
    plt.show()
    
def test_false_positives():
    n_vehicles = 2800
    sensor_reads = 360
    x_values = [0.0, 0.15, 0.3, 0.65, 1.0]
    y_values = []
    for fpr in x_values:
        y_values.append(sum([
                simulate_false_positives(
                    vehicles_per_hour=n_vehicles,
                    sensor_reads_per_hour=sensor_reads,
                    fpr=fpr
                ) for _ in range(sensor_reads)
            ]
            ) / n_vehicles
        )
    plt.plot(x_values, y_values, ".-", label="False Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Number of Vehicles Detected")
    plt.title("False Positive Rate vs. Number of Vehicles Detected")
    plt.legend(loc='best')
    plt.grid(color="#dddddd")
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    test_true_positives()
    test_false_positives()