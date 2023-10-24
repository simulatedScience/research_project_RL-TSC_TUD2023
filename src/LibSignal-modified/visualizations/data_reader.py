"""
A program to read the `[datetime]_DTL.log` files generated during training of RL TSC agents using the LibSignal library.

Author: Sebastian Jost & GPT-4 (19.10.2023)
"""

import csv

# Function to read the data file into separate lists
def read_rl_training_data_single_loop(file_path: str):
    """
    Reads the RL TSC agent output log file and returns the data as separate lists for each column.
    This function uses a single loop to read the data and populate the dictionaries.

    Parameters:
        file_path (str): The path to the RL TSC agent output log file

    Returns:
        train_records (dict): Dictionary containing training records
        test_records (dict): Dictionary containing test records
    """
    # Initialize dictionaries for storing training and test records
    train_records = {
        "episode": [],
        "real_avg_travel_time": [],
        "q_loss": [],
        "rewards": [],
        "queue": [],
        "delay": [],
        "throughput": []
    }

    test_records = {
        "episode": [],
        "real_avg_travel_time": [],
        "q_loss": [],
        "rewards": [],
        "queue": [],
        "delay": [],
        "throughput": []
    }

    # Read the file and populate the dictionaries
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            record_type = row[1]  # "TRAIN" or "TEST"
            episode = int(row[2])
            real_avg_travel_time = float(row[3])
            q_loss = float(row[4])
            rewards = float(row[5])
            queue = float(row[6])
            delay = float(row[7])
            throughput = int(row[8])

            if record_type == "TRAIN":
                train_records["episode"].append(episode)
                train_records["real_avg_travel_time"].append(real_avg_travel_time)
                train_records["q_loss"].append(q_loss)
                train_records["rewards"].append(rewards)
                train_records["queue"].append(queue)
                train_records["delay"].append(delay)
                train_records["throughput"].append(throughput)
            elif record_type == "TEST":
                test_records["episode"].append(episode)
                test_records["real_avg_travel_time"].append(real_avg_travel_time)
                test_records["q_loss"].append(q_loss)
                test_records["rewards"].append(rewards)
                test_records["queue"].append(queue)
                test_records["delay"].append(delay)
                test_records["throughput"].append(throughput)

    return train_records, test_records