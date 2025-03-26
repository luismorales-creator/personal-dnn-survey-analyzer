import pandas as pd
import re
import os

def extract_participant_number(code):
    """Extract the participant number from a participation code."""
    if not isinstance(code, str):
        return None
    
    # Try to extract a number from the code
    match = re.search(r'[sS](\d+)', code)
    if match:
        return int(match.group(1))
    
    # Special case for "So9" -> 9
    if code.lower() == 'so9bdb1':
        return 9
    
    return None

def identify_duplicates(df):
    """Identify duplicate participants based on extracted numbers."""
    # Extract participant numbers
    df['Participant_Number'] = df['Participation Code'].apply(extract_participant_number)
    
    # Group by participant number and find duplicates
    duplicates = {}
    for num, group in df.groupby('Participant_Number'):
        if num is not None and len(group) > 1:
            duplicates[num] = group['Participation Code'].tolist()
    
    return duplicates

def merge_type1_duplicates(df, duplicates, known_type2=[13, 15, 16, 25]):
    """Merge Type 1 duplicates (different capitalization/formatting)."""
    # Create a mapping from original codes to standardized codes
    code_mapping = {}
    
    for num, codes in duplicates.items():
        # Skip known Type 2 duplicates
        if num in known_type2:
            continue
        
        # Choose the most standardized version (prefer uppercase S format)
        preferred_code = None
        for code in codes:
            if code.startswith('S') and (preferred_code is None or len(code) < len(preferred_code)):
                preferred_code = code
        
        if preferred_code is None:
            # If no S-prefixed code, use the shortest one
            preferred_code = min(codes, key=len)
        
        # Map all variations to the preferred code
        for code in codes:
            code_mapping[code] = preferred_code
    
    # Apply the mapping to create a new dataframe
    df_merged = df.copy()
    df_merged['Original_Code'] = df_merged['Participation Code']
    df_merged['Participation Code'] = df_merged['Participation Code'].map(
        lambda x: code_mapping.get(x, x))
    
    # Group by the new participation codes and take the first row for each group
    # (this effectively merges the Type 1 duplicates)
    df_merged = df_merged.groupby('Participation Code', as_index=False).first()
    
    return df_merged, code_mapping

def handle_type2_duplicates(df, known_type2=[13, 15, 16, 25]):
    """Handle Type 2 duplicates (same survey filled out twice)."""
    result_df = df.copy()
    
    # Process each known Type 2 duplicate
    for num in known_type2:
        # Find rows for this participant
        participant_rows = result_df[result_df['Participant_Number'] == num]
        
        print(f"\nProcessing Type 2 duplicate for participant {num}:")
        print(f"  Found {len(participant_rows)} rows")
        
        if len(participant_rows) < 2:
            print(f"  Skipping participant {num} - not enough rows")
            continue  # Skip if there's only one row
        
        # Print the participation codes for debugging
        codes = participant_rows['Participation Code'].tolist()
        print(f"  Participation codes: {codes}")
        
        # Create a new row that combines data from both rows
        combined_row = participant_rows.iloc[0].copy()  # Start with the first row
        
        # Use session 1 data from the second row as session 2 data in the combined row
        second_row = participant_rows.iloc[1]
        
        # Map session 1 columns from second row to session 2 columns in combined row
        column_mapping = {
            'STAI-S Total_Session1': 'STAI-S Total_Session2',
            'PANAS Positive Total_Session1': 'PANAS Positive Total_Session2',
            'PANAS Negative Total_Session1': 'PANAS Negative Total_Session2'
        }
        
        for s1_col, s2_col in column_mapping.items():
            if s1_col in second_row.index and pd.notna(second_row[s1_col]):
                combined_row[s2_col] = second_row[s1_col]
                print(f"  Mapped {s1_col} ({second_row[s1_col]}) to {s2_col}")
        
        # Remove all rows for this participant
        result_df = result_df[result_df['Participant_Number'] != num]
        
        # Add the combined row
        result_df = pd.concat([result_df, pd.DataFrame([combined_row])], ignore_index=True)
        
        print(f"  Successfully combined rows for participant {num}")
    
    return result_df

def main():
    # File paths
    input_file = "output/summary_file.csv"
    output_file = "output/summary_file_fixed.csv"
    
    # Read the summary file
    df = pd.read_csv(input_file)
    
    # Extract participant numbers and identify duplicates
    duplicates = identify_duplicates(df)
    
    # Print the duplicates for reference
    print("Identified duplicate participant codes:")
    for num, codes in duplicates.items():
        print(f"  Participant {num}: {codes}")
    
    # Merge Type 1 duplicates
    df_merged, code_mapping = merge_type1_duplicates(df, duplicates)
    
    # Print the mapping for reference
    print("\nStandardized participation codes:")
    for orig, std in code_mapping.items():
        print(f"  {orig} → {std}")
    
    # Handle Type 2 duplicates
    df_final = handle_type2_duplicates(df_merged)
    
    # Sort by participant number for better readability
    if 'Participant_Number' in df_final.columns:
        df_final = df_final.sort_values('Participant_Number')
    
    # Remove the temporary columns
    if 'Participant_Number' in df_final.columns:
        df_final = df_final.drop(columns=['Participant_Number'])
    if 'Original_Code' in df_final.columns:
        df_final = df_final.drop(columns=['Original_Code'])
    
    # Save the fixed summary file
    df_final.to_csv(output_file, index=False)
    print(f"\nFixed summary file saved to {output_file}")

if __name__ == "__main__":
    main()
