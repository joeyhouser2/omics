#split excel file

import pandas as pd
import os

INPUT_FILE = 'C:/Users/joeyh/Downloads/LinkedOmics_Combined.xlsx'

OUTPUT_FOLDER = 'C:/Users/joeyh/Documents/GitHub/omics/data'
# -----------------------------------------------

def main():
    # Check if file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find '{INPUT_FILE}'.")
        print("Make sure this script is in the same folder as your Excel file.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder: {OUTPUT_FOLDER}")

    print("Reading Excel file... (this might take a moment)")
    
    # Load the Excel file object (this is faster than reading all data at once)
    try:
        xls = pd.ExcelFile(INPUT_FILE)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Iterate through every sheet
    for sheet_name in xls.sheet_names:
        print(f"Extracting sheet: {sheet_name}...")
        
        try:
            # Read the specific sheet
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Construct the output filename
            # We replace spaces with underscores to make filenames cleaner
            clean_name = sheet_name.replace(' ', '_')
            output_filename = f"{clean_name}.csv"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"Failed to extract {sheet_name}: {e}")

    print(f"\nSuccess! All CSV files are in the '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    main()