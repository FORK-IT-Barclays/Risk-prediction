import pandas as pd
import os

def load_berka_dataset(data_dir: str):
    """
    Loads the PKDD99 Czech Republic (Berka) dataset files from the specified directory.
    Returns a dictionary of pandas DataFrames.
    """
    datasets = {}
    
    # List of all files in the Berka dataset
    files = [
        "account.csv", "card.csv", "client.csv", "disp.csv", 
        "district.csv", "loan.csv", "order.csv", "trans.csv"
    ]
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"Loading {file}...")
            # The dataset uses semicolons as delimiters
            # low_memory=False is helpful for large files like trans.csv
            df = pd.read_csv(file_path, sep=";", low_memory=False)
            
            # Use the filename without extension as the dictionary key
            key_name = file.split(".")[0]
            datasets[key_name] = df
        else:
            print(f"Warning: {file} not found at {file_path}.")
            
    return datasets

if __name__ == "__main__":
    # Base directory for the dataset
    dataset_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    
    if not os.path.exists(dataset_directory):
        print(f"Error: The directory {dataset_directory} does not exist.")
    else:
        print("Starting to load the Berka dataset...")
        data = load_berka_dataset(dataset_directory)
        
        print("\nDataset loaded successfully!")
        print(f"Loaded tables: {list(data.keys())}")
        
        # Display an example preview
        if "account" in data and "loan" in data:
            print("\n--- Account Table Preview ---")
            print(data["account"].head())
            print(f"\nAccount table shape: {data['account'].shape}")

            print("\n--- Loan Table Preview ---")
            print(data['loan'].head())
            print(f"\nLoan table shape: {data['loan'].shape}")

            print("\n--- Client Table Preview ---")
            print(data['client'].head())
            print(f"\nClient table shape: {data['client'].shape}")

            print("\n--- Disp Table Preview ---")
            print(data['disp'].head())
            print(f"\nDisp table shape: {data['disp'].shape}")

            print("\n--- District Table Preview ---")
            print(data['district'].head())
            print(f"\nDistrict table shape: {data['district'].shape}")

            print("\n--- Order Table Preview ---")
            print(data['order'].head())
            print(f"\nOrder table shape: {data['order'].shape}")

            print("\n--- Trans Table Preview ---")
            print(data['trans'].head())
            print(f"\nTrans table shape: {data['trans'].shape}")

            print("\n--- Card Table Preview ---")
            print(data['card'].head())
            print(f"\nCard table shape: {data['card'].shape}")

            # JOINING: Loan and Account tables based on account_id
            print("\n--- Joined Loan and Account Table ---")
            loan_account_df = pd.merge(data["loan"], data["account"], on="account_id", how="inner")
            print(f"Merged table shape: {loan_account_df.shape}")
            print(loan_account_df.head())
