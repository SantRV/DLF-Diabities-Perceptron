
import os
import pandas as pd
from pathlib import Path
import re


class Utils():
    def __init__(self) -> None:
        pass

    # Define a function to parse the text line into class and features
    def parse_line(self, line):
        parts = line.strip().split()
        class_label = int(parts[0])
        features = {}
        for part in parts[1:]:
            match = re.match(r'(\d+):([\d.]+)', part)
            if match:
                feature_id, feature_value = map(float, match.groups())
                features[int(feature_id)] = feature_value
        return {"class": class_label, **features}

    def load_data_txt(self, file_name: str) -> pd.DataFrame:
       # Construct the full file path
        parsed_path = os.path.join(
            Path().resolve(), "data", file_name)

        # Read the text file into a list of lines
        data_list = []
        with open(parsed_path, 'r') as file:
            for line in file:
                data_list.append(self.parse_line(line))

        # Convert the list of dictionaries into a pandas DataFrame
        df = pd.DataFrame(data_list)

        return df

    def load_data_csv(self, file_name: str) -> pd.DataFrame:
       # Construct the full file path
        parsed_path = os.path.join(
            Path().resolve().parent, "data", file_name)

        df = pd.read_csv(parsed_path)

        return df
