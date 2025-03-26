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
    # Read the summary file
    df = pd.read_csv(input_file)
    
    # Extract participant numbers
    df['Participant_Number'] = df['Participation Code'].apply(extract_participant_number)
    
    # Identify duplicates
    duplicates = {}
    for num, group in df.groupby('Participant_Number'):
        if num is not None and len(group) > 1:
            duplicates[num] = group.index.tolist()
    
    # Print the duplicates for reference
    print(f"Found {len(duplicates)} participants with duplicate entries:")
    for num, indices in duplicates.items():
        codes = df.loc[indices, 'Participation Code'].tolist()
        print(f"  Participant {num}: {codes}")
    
    # Create a new dataframe for the fixed data
    fixed_df = df.copy()
    
    # Process each duplicate
    for num, indices in duplicates.items():
        if len(indices) < 2:
            continue
        
        # Get the rows for this participant
        rows = df.loc[indices].copy()
        
        # Standardize the participation code (prefer uppercase S format)
        codes = rows['Participation Code'].tolist()
        preferred_code = None
        for code in codes:
            if isinstance(code, str) and code.startswith('S') and (preferred_code is None or len(code) < len(preferred_code)):
                preferred_code = code
        
        if preferred_code is None:
            # If no S-prefixed code, use the shortest one
            preferred_code = min([c for c in codes if isinstance(c, str)], key=len, default=codes[0])
        
        # Create a combined row starting with the first row
        combined_row = rows.iloc[0].copy()
        combined_row['Participation Code'] = preferred_code
        
        # If there's a second row, use its session 1 data as session 2 data in the combined row
        if len(rows) > 1:
            second_row = rows.iloc[1]
            
            # Map session 1 columns from second row to session 2 columns in combined row
            column_mapping = {
                'STAI-S Total_Session1': 'STAI-S Total_Session2',
                'PANAS Positive Total_Session1': 'PANAS Positive Total_Session2',
                'PANAS Negative Total_Session1': 'PANAS Negative Total_Session2'
            }
            
            for s1_col, s2_col in column_mapping.items():
                if s1_col in second_row.index and pd.notna(second_row[s1_col]):
                    combined_row[s2_col] = second_row[s1_col]
                    print(f"  Mapped {s1_col} ({second_row[s1_col]}) to {s2_col} for participant {num}")
        
        # Remove all rows for this participant from the fixed dataframe
        fixed_df = fixed_df[fixed_df['Participant_Number'] != num]
        
        # Add the combined row
        fixed_df = pd.concat([fixed_df, pd.DataFrame([combined_row])], ignore_index=True)
        
        print(f"  Successfully combined rows for participant {num}")
    
    # Sort by participant number for better readability
    if 'Participant_Number' in fixed_df.columns:
        fixed_df = fixed_df.sort_values('Participant_Number')
    
    # Remove the temporary column
    if 'Participant_Number' in fixed_df.columns:
        fixed_df = fixed_df.drop(columns=['Participant_Number'])
    
    # Save the fixed summary file
    fixed_df.to_csv(output_file, index=False)
    print(f"\nFixed summary file saved to {output_file}")
    
    return fixed_df

if __name__ == "__main__":
    input_file = "output/summary_file.csv"
    output_file = "output/summary_file_fixed.csv"
    fix_duplicates(input_file, output_file)
