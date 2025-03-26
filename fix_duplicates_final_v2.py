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
            duplicates[num] = group.index.tolist()
    
    return duplicates

def merge_type1_duplicates(df, duplicates, known_type2=[13, 15, 16, 25]):
    """Merge Type 1 duplicates (different capitalization/formatting)."""
    # Create a mapping from original codes to standardized codes
    code_mapping = {}
    
    for num, indices in duplicates.items():
        # Skip known Type 2 duplicates
        if num in known_type2:
            continue
        
        # Get the participation codes for this participant
        codes = df.loc[indices, 'Participation Code'].tolist()
        
        # Choose the most standardized version (prefer uppercase S format)
        preferred_code = None
        for code in codes:
            if isinstance(code, str) and code.startswith('S') and (preferred_code is None or len(code) < len(preferred_code)):
                preferred_code = code
        
        if preferred_code is None:
            # If no S-prefixed code, use the shortest one
            preferred_code = min([c for c in codes if isinstance(c, str)], key=len, default=codes[0])
        
        # Map all variations to the preferred code
        for code in codes:
            if isinstance(code, str):
                code_mapping[code] = preferred_code
    
    # Apply the mapping to create a new dataframe
    df_merged = df.copy()
    df_merged['Original_Code'] = df_merged['Participation Code']
    df_merged['Participation Code'] = df_merged['Participation Code'].map(
        lambda x: code_mapping.get(x, x) if isinstance(x, str) else x)
    
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
        
        # For Type 2 duplicates, we want to:
        # 1. Use the earlier response as the base row
        # 2. Take the session 1 data from the later response and use it as session 2 data
        
        # Since we don't have date information, we'll use the row order as a proxy
        # Assuming the first row is the earlier response
        earlier_row = participant_rows.iloc[0]
        later_row = participant_rows.iloc[1]
        
        # Create a combined row starting with the earlier row
        combined_row = earlier_row.copy()
        
        # Standardize the participation code (prefer uppercase S format)
        preferred_code = None
        for code in codes:
            if isinstance(code, str) and code.startswith('S') and (preferred_code is None or len(code) < len(preferred_code)):
                preferred_code = code
        
        if preferred_code is None:
            # If no S-prefixed code, use the shortest one
            preferred_code = min([c for c in codes if isinstance(c, str)], key=len, default=codes[0])
        
        combined_row['Participation Code'] = preferred_code
        
        # Map session 1 data from later row to session 2 data in combined row
        column_mapping = {
            'STAI-S Total_Session1': 'STAI-S Total_Session2',
            'PANAS Positive Total_Session1': 'PANAS Positive Total_Session2',
            'PANAS Negative Total_Session1': 'PANAS Negative Total_Session2'
        }
        
        for s1_col, s2_col in column_mapping.items():
            if s1_col in later_row.index and pd.notna(later_row[s1_col]):
                combined_row[s2_col] = later_row[s1_col]
                print(f"  Mapped {s1_col} ({later_row[s1_col]}) to {s2_col}")
        
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
    
    # Save the original columns
    original_columns = df.columns.tolist()
    
    # Make a working copy of the dataframe
    working_df = df.copy()
    
    # Extract participant numbers and identify duplicates
    duplicates = identify_duplicates(working_df)
    
    # Print the duplicates for reference
    print("Identified duplicate participant codes:")
    for num, indices in duplicates.items():
        codes = working_df.loc[indices, 'Participation Code'].tolist()
        print(f"  Participant {num}: {codes}")
    
    # Handle Type 2 duplicates first
    df_type2_handled = handle_type2_duplicates(working_df)
    
    # Identify remaining duplicates (should only be Type 1 now)
    remaining_duplicates = identify_duplicates(df_type2_handled)
    
    # Remove known Type 2 duplicates from the remaining duplicates
    known_type2 = [13, 15, 16, 25]
    for num in known_type2:
        if num in remaining_duplicates:
            del remaining_duplicates[num]
    
    # Merge Type 1 duplicates
    df_merged, code_mapping = merge_type1_duplicates(df_type2_handled, remaining_duplicates, known_type2=[])
    
    # Print the mapping for reference
    print("\nStandardized participation codes:")
    for orig, std in code_mapping.items():
        print(f"  {orig} → {std}")
    
    # Use the merged dataframe as the final result
    df_final = df_merged
    
    # Sort by participant number for better readability
    if 'Participant_Number' in df_final.columns:
        df_final = df_final.sort_values('Participant_Number')
    
    # Remove the temporary columns
    if 'Participant_Number' in df_final.columns:
        df_final = df_final.drop(columns=['Participant_Number'])
    if 'Original_Code' in df_final.columns:
        df_final = df_final.drop(columns=['Original_Code'])
    
    # Get the original columns from the input file
    original_columns = df.columns.tolist()
    
    # Ensure the columns match exactly those in the input file
    df_final = df_final[original_columns]
    
    # Save the fixed summary file
    df_final.to_csv(output_file, index=False)
    print(f"\nFixed summary file saved to {output_file}")

if __name__ == "__main__":
    main()
