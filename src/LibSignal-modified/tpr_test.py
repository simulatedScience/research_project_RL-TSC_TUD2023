from generator.lane_vehicle import simulate_true_positives, simulate_false_positives

# create plot showing true positive rate vs. number of vehicles detected
import matplotlib.pyplot as plt

def test_true_positives(n_vehicles: int=1000):
    true_tpr_values = [0.0, 0.3, 0.6, 0.8, 0.95, 1.0]
    measured_tpr_values = []
    for tpr in true_tpr_values:
        # simulate episode by simulating detection of `n_vehicles` vehicles with true positive rate `tpr`
        # calculate measured tpr as the number of true positives divided by the true number of vehicles
        measured_tpr_values.append(simulate_true_positives(n_vehicles, tpr)/n_vehicles)
    plt.plot(true_tpr_values, measured_tpr_values, ".-", label="True Positive Rate")
    plt.xlabel("Expected True Positive Rate")
    plt.ylabel("Measured True Positive Rate")
    plt.title("Measured vs. Expected True Positive Rate")
    plt.legend(loc='best')
    plt.grid(color="#dddddd")
    plt.tight_layout()
    plt.show()
    
def test_false_positives(n_vehicles: int = 2800, sensor_reads: int = 360):
    true_tpr_values = [0.3, 0.6, 0.8, 0.95, 1.0]
    true_fpr_values = [0.0, 0.15, 0.3, 0.65, 0.95]
    measured_fpr_values: dict = {} # key: tpr, value: measured fpr
    for tpr in true_tpr_values:
        measured_fpr_values[tpr]: list[int] = []
        for fpr in true_fpr_values:
            # simulate episode by simulating `sensor_reads` number of sensor reads with false positive rate `fpr`
            # calculate measured fpr as the number of false positives divided by the measured number of vehicles
            false_positives: int = sum([
                    simulate_false_positives(
                        vehicles_per_hour=n_vehicles,
                        sensor_reads_per_hour=sensor_reads,
                        fpr=fpr,
                        tpr=tpr,
                    ) for _ in range(sensor_reads)
                ]
            )
            true_positives = simulate_true_positives(n_vehicles, tpr)
            measured_vehicles = true_positives + false_positives
            measured_fpr_values[tpr].append(
                false_positives / measured_vehicles
            )
            print(f"tpr: {tpr}, \tfpr: {fpr}, \ttrue_positives: {true_positives:.0f}, \tfalse_positives: {false_positives:.0f}, \tmeasured_vehicles: {measured_vehicles:.0f}")

    for tpr, measured_fpr in measured_fpr_values.items():
        # map tpr to hue
        n_tprs = len(true_tpr_values)
        linecolor = plt.get_cmap('hsv')(1/n_tprs + n_tprs * tpr / (n_tprs+1))
        plt.plot(true_fpr_values, measured_fpr, ".-", label=f"measured FPR for TPR = {tpr}", color=linecolor)
    plt.xlabel("expected False Positive Rate")
    plt.ylabel("measured False Positive Rate")
    plt.title("Measured vs. Expected False Positive Rate")
    plt.legend(loc='best')
    plt.grid(color="#dddddd")
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    n_vehicles = 28000
    test_true_positives(n_vehicles=n_vehicles)
    test_false_positives(n_vehicles=n_vehicles)