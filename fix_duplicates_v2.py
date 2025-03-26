import pandas as pd
import re

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

def fix_duplicates(input_file, output_file):
    """Fix duplicate participant entries in the summary file."""
    # Read the original summary file
    df = pd.read_csv(input_file)
    
    # Make a copy to work with
    result_df = df.copy()
    
    # Extract participant numbers
    result_df['Participant_Number'] = result_df['Participation Code'].apply(extract_participant_number)
    
    # Identify duplicates
    duplicates = []
    for num, group in result_df.groupby('Participant_Number'):
        if num is not None and len(group) > 1:
            duplicates.append(num)
    
    # Print the duplicates for reference
    print(f"Found {len(duplicates)} participants with duplicate entries:")
    for num in duplicates:
        group = result_df[result_df['Participant_Number'] == num]
        codes = group['Participation Code'].tolist()
        print(f"  Participant {num}: {codes}")
    
    # Process each duplicate
    for num in duplicates:
        # Get the rows for this participant
        rows = result_df[result_df['Participant_Number'] == num].copy()
        
        if len(rows) < 2:
            continue
        
        # For participants with session 1 surveys filled out twice, we want to:
        # 1. Keep the row with the most complete data as the base row
        # 2. Use the session 1 data from the other row as session 2 data in the combined row
        
        # Choose the row with the most non-null values as the base row
        non_null_counts = rows.count(axis=1)
        base_row_idx = non_null_counts.idxmax()
        
        # Get the other rows
        other_rows = rows[rows.index != base_row_idx]
        if len(other_rows) == 0:
            continue
        
        # Use the first other row for mapping
        other_row_idx = other_rows.index[0]
        
        # Get the base row and other row
        base_row = rows.loc[base_row_idx]
        other_row = rows.loc[other_row_idx]
        
        # Standardize the participation code (prefer uppercase S format)
        codes = rows['Participation Code'].tolist()
        preferred_code = None
        for code in codes:
            if isinstance(code, str) and code.startswith('S') and (preferred_code is None or len(code) < len(preferred_code)):
                preferred_code = code
        
        if preferred_code is None:
            # If no S-prefixed code, use the shortest one
            preferred_code = min([c for c in codes if isinstance(c, str)], key=len, default=codes[0])
        
        # Create a combined row starting with the base row
        combined_row = base_row.copy()
        combined_row['Participation Code'] = preferred_code
        
        print(f"\nProcessing duplicate for participant {num}:")
        print(f"  Base row: {base_row['Participation Code']}")
        print(f"  Other row: {other_row['Participation Code']}")
        print(f"  Preferred code: {preferred_code}")
        
        # Map session 1 data from other row to session 2 data in combined row
        column_mapping = {
            'STAI-S Total_Session1': 'STAI-S Total_Session2',
            'PANAS Positive Total_Session1': 'PANAS Positive Total_Session2',
            'PANAS Negative Total_Session1': 'PANAS Negative Total_Session2'
        }
        
        for s1_col, s2_col in column_mapping.items():
            # Only map if the target column is empty in the base row
            if s1_col in other_row.index and pd.notna(other_row[s1_col]) and (s2_col not in base_row.index or pd.isna(base_row[s2_col])):
                combined_row[s2_col] = other_row[s1_col]
                print(f"  Mapped {s1_col} ({other_row[s1_col]}) to {s2_col}")
        
        # Remove all rows for this participant from the result dataframe
        result_df = result_df[result_df['Participant_Number'] != num]
        
        # Add the combined row
        result_df = pd.concat([result_df, pd.DataFrame([combined_row])], ignore_index=True)
        
        print(f"  Successfully combined rows for participant {num}")
    
    # Sort by participant number for better readability
    if 'Participant_Number' in result_df.columns:
        result_df = result_df.sort_values('Participant_Number')
    
    # Remove the temporary column
    if 'Participant_Number' in result_df.columns:
        result_df = result_df.drop(columns=['Participant_Number'])
    
    # Ensure the columns match exactly those in the input file
    result_df = result_df[df.columns]
    
    # Save the fixed summary file
    result_df.to_csv(output_file, index=False)
    print(f"\nFixed summary file saved to {output_file}")
    
    return result_df

if __name__ == "__main__":
    fix_duplicates("output/summary_file.csv", "output/summary_file_fixed.csv")
