import os

def extract_zinc_smiles(file_path):
    zinc_smiles = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # Ensure the line is not empty
                parts = line.split('\t')  # Assuming tab-delimited file
                if len(parts) >= 2:  # At least SMILES and Zinc ID
                    smiles = parts[0].strip()
                    zinc_id = parts[1].strip()
                    zinc_smiles.append((smiles, zinc_id))

    return zinc_smiles

def collect_data_from_folders(root_folder):
    all_zinc_smiles = []

    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    all_zinc_smiles.extend(extract_zinc_smiles(file_path))

    return all_zinc_smiles

def save_to_txt(data, output_file):
    """
    Save the extracted data to a tab-delimited .txt file.

    Args:
        data (list): List of tuples containing SMILES and Zinc IDs.
        output_file (str): The output .txt file path.
    """
    with open(output_file, 'w') as txtfile:
        for smiles, zinc_id in data:
            txtfile.write(f"{smiles}\t{zinc_id}\n")

def main():
    root_folder = '.'  # Set this to your root folder if different
    output_file = 'zinc_smiles.txt'  # Tab-delimited .txt output

    data = collect_data_from_folders(root_folder)
    save_to_txt(data, output_file)
    print(f"Data successfully saved to {output_file}")

if __name__ == "__main__":
    main()
