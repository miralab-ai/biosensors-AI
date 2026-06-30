import os
import pandas as pd
import numpy as np
import pickle

def load_raw_excel_data(data_folder="data/lab_measurements"):
    """
    Reads all Excel files in the raw measurements directory and extracts the DPV current values.
    Returns a 3D list: [class_index][measurement_index][voltage_points]
    """
    all_data = []
    
    # Check if the folder exists
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Please place your raw Excel measurements in the '{data_folder}' directory.")
        
    class_folders = sorted([d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))])
    
    for class_folder in class_folders:
        class_path = os.path.join(data_folder, class_folder)
        class_data = []
        
        for file_name in sorted(os.listdir(class_path)):
            if file_name.endswith(('.xlsx', '.xls')):
                file_path = os.path.join(class_path, file_name)
                
                # Read the Excel file, skip the metadata row (first row)
                data = pd.read_excel(file_path)
                selected_column = data.iloc[1:, 1].dropna().tolist()
                class_data.append(selected_column)
                
        all_data.append(class_data)
        
    return all_data, class_folders

def create_pickle_datasets(all_data, num_signals=5):
    """Applies convex combination and returns raw & augmented dictionaries."""
    dict_augmented = {}
    dict_raw = {}
    
    for cls_idx, class_data in enumerate(all_data):
        interpolated_signals = []
        raw_signals = []
        df = pd.DataFrame(class_data)
        
        start_val = 1  
        for first_cycle_idx in range(0, len(df) - 1):
            for second_cycle_idx in range(start_val, len(df)):
                signal1 = df.iloc[first_cycle_idx]
                signal2 = df.iloc[second_cycle_idx]
                
                for i in range(1, num_signals + 1):
                    alpha = i / (num_signals + 1)
                    new_signal = (1 - alpha) * signal1 + alpha * signal2
                    interpolated_signals.append(list(new_signal))
            start_val += 1
            
        for i in range(len(df)):
            raw_signals.append(list(df.iloc[i]))
            interpolated_signals.append(list(df.iloc[i]))

        class_key = f"class_{cls_idx}"
        dict_augmented[class_key] = interpolated_signals
        dict_raw[class_key] = raw_signals
        
    return dict_augmented, dict_raw

if __name__ == "__main__":
    print("Step 1: Reading raw data from 'data/lab_measurements' directory...")
    all_data, class_names = load_raw_excel_data("data/lab_measurements")
    print(f"A total of {len(class_names)} classes were found.")
    
    print("Step 2: Applying interpolation algorithm (Data Augmentation)...")
    dict_augmented, dict_raw = create_pickle_datasets(all_data, num_signals=5)
    
    # Create output directory
    output_dir = "data/processed_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Step 3: Saving '.pkl' files into '{output_dir}' directory...")
    with open(f'{output_dir}/data_augmented.pkl', 'wb') as file:
        pickle.dump(dict_augmented, file)
        
    with open(f'{output_dir}/data_raw.pkl', 'wb') as file:
        pickle.dump(dict_raw, file)
        
    print("Process completed successfully! Datasets are ready for training.")