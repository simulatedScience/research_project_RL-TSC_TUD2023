"""
A program to read the `[datetime]_DTL.log` files generated during training of RL TSC agents using the LibSignal library.

Author: Sebastian Jost & GPT-4 (19.10.2023)
"""

from collections import namedtuple
import csv

# Define the named tuple
NoiseSettings = namedtuple('NoiseSettings', ['failure_chance', 'true_positive_rate', 'false_positive_rate'])


def read_and_group_test_data(filepath: str) -> dict:
    """
    Read the test data from the provided file and group by noise settings.
    
    Args:
    - filepath (str): Path to the file containing test data.
    
    Returns:
    - dict: A dictionary grouping data by noise settings.
    """
    data_list = read_test_data(filepath)
    
    # Dictionary to group runs by noise settings
    grouped_data = {}
    
    for run in data_list:
        key = NoiseSettings(run['failure chance'], run['true positive rate'], run['false positive rate'])
        
        if key not in grouped_data:
            grouped_data[key] = {
                'travel_time': [],
                'mean_rewards': [],
                'queue': [],
                'delay': [],
                'throughput': []
            }
        
        # Append metrics to the group
        for metric in ['travel_time', 'mean_rewards', 'queue', 'delay', 'throughput']:
            grouped_data[key][metric].append(run[metric])
    
    return grouped_data


def read_test_data(filepath: str) -> list:
    """
    Read the test data from the provided file and return as a list of dictionaries.
    
    Args:
    - filepath (str): Path to the file containing test data.
    
    Returns:
    - list: A list of dictionaries, each containing settings and metrics for a test run.
    """
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Split the content into blocks
    blocks = content.strip().split("Running RL Experiment")
    
    # Initialize a list to hold extracted data
    data_list = []
    
    for block in blocks:
        lines = block.strip().split("\n")
        
        # Skip blocks without the required number of lines
        if len(lines) < 2:
            continue
        
        run_info = extract_settings(lines[1])
        run_info.update(extract_metrics(lines[2]))
        data_list.append(run_info)
    
    return data_list


def extract_settings(line: str) -> dict:
    """
    Extract settings (fc, tpr, fpr, rep) from the given line and return as a dictionary.
    
    Args:
    - line (str): The line containing settings information.
    
    Returns:
    - dict: A dictionary containing the extracted settings.
    """
    settings = {
        'failure chance': float(line.split("fc=")[1].split("_")[0]),
        'true positive rate': float(line.split("tpr=")[1].split("_")[0]),
        'false positive rate': float(line.split("fpr=")[1]), # line ends after fpr value
        'rep': int(line.split("id=")[1].strip("rep_").split("_")[0])
    }
    return settings


def extract_metrics(line: str) -> dict:
    """
    Extract metrics from the given line and return as a dictionary.
    
    Args:
    - line (str): The line containing metrics information.
    
    Returns:
    - dict: A dictionary containing the extracted metrics.
    """
    metrics_parts = line.split(", ")
    metrics = {
        'travel_time': float(metrics_parts[0].split("is ")[1]),
        'mean_rewards': float(metrics_parts[1].split(": ")[1]),
        'queue': float(metrics_parts[2].split(": ")[1]),
        'delay': float(metrics_parts[3].split(": ")[1]),
        'throughput': int(metrics_parts[4].split(": ")[1])
    }
    return metrics


def read_rl_training_data(file_path: str):
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

if __name__ == "__main__":
    # prompt user to enter a file via filedialog
    import tkinter as tk
    from tkinter import filedialog
    # read test file
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    # read rl training file
    grouped_test_data = read_and_group_test_data(file_path)
    # display data
    print(grouped_test_data)
