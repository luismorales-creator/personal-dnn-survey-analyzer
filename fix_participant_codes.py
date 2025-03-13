import pandas as pd
import re
import os

def normalize_code(code):
    """Normalize a participation code for fuzzy matching."""
    if not isinstance(code, str):
        return ""
    
    # Convert to lowercase and strip whitespace
    normalized = code.lower().strip()
    
    # Remove any 'pdb' suffix variations
    normalized = re.sub(r'pdb\d*', '', normalized)
    
    # Remove any non-alphanumeric characters
    normalized = re.sub(r'[^a-z0-9]', '', normalized)
    
    return normalized

def find_potential_duplicates(summary_file):
    """Find potential duplicate participation codes in the summary file."""
    # Read the summary file
    df = pd.read_csv(summary_file)
    
    # Create a normalized version of the participation codes
    df['Normalized Code'] = df['Participation Code'].apply(normalize_code)
    
    # Group by normalized code and find groups with more than one unique original code
    potential_duplicates = {}
    for norm_code, group in df.groupby('Normalized Code'):
        if len(group['Participation Code'].unique()) > 1:
            potential_duplicates[norm_code] = group['Participation Code'].unique().tolist()
    
    return potential_duplicates

def generate_mapping_file(potential_duplicates, output_file):
    """Generate a mapping file for the user to review and edit."""
    # Create a DataFrame with columns for normalized code, original codes, and suggested standardized code
    rows = []
    for norm_code, orig_codes in potential_duplicates.items():
        # Choose the most standardized version (prefer uppercase S format)
        suggested_code = None
        for code in orig_codes:
            if code.startswith('S') and (suggested_code is None or len(code) < len(suggested_code)):
                suggested_code = code
        
        if suggested_code is None:
            # If no S-prefixed code, use the shortest one
            suggested_code = min(orig_codes, key=len)
        
        # Add a row for each original code
        for orig_code in orig_codes:
            rows.append({
                'Normalized Code': norm_code,
                'Original Code': orig_code,
                'Standardized Code': suggested_code,
                'Keep Separate': 'No'  # Default to merging
            })
    
    # Create the DataFrame and save to CSV
    mapping_df = pd.DataFrame(rows)
    mapping_df.to_csv(output_file, index=False)
    
    return mapping_df

def apply_mapping(summary_file, mapping_file, output_file):
    """Apply the mapping to create a corrected summary file."""
    # Read the summary file and mapping file
    df = pd.read_csv(summary_file)
    mapping_df = pd.read_csv(mapping_file)
    
    # Create a dictionary mapping original codes to standardized codes
    # Only for rows where Keep Separate is 'No'
    code_mapping = {}
    for _, row in mapping_df.iterrows():
        if row['Keep Separate'].lower() == 'no':
            code_mapping[row['Original Code']] = row['Standardized Code']
    
    # Apply the mapping to the Participation Code column
    df['Participation Code'] = df['Participation Code'].map(lambda x: code_mapping.get(x, x))
    
    # Group by Participation Code and merge the data
    # This is a simplified approach - in reality, you might need more complex merging logic
    merged_df = df.groupby('Participation Code', as_index=False).first()
    
    # Save the corrected summary file
    merged_df.to_csv(output_file, index=False)
    
    return merged_df

def main():
    # File paths
    summary_file = "summary_file.csv"
    mapping_file = "participant_code_mapping.csv"
    corrected_file = "summary_file_corrected.csv"
    
    # Step 1: Find potential duplicates
    potential_duplicates = find_potential_duplicates(summary_file)
    print(f"Found {len(potential_duplicates)} normalized codes with multiple variations")
    
    # Step 2: Generate mapping file
    if potential_duplicates:
        mapping_df = generate_mapping_file(potential_duplicates, mapping_file)
        print(f"Generated mapping file: {mapping_file}")
        print("Please review and edit this file:")
        print("- Verify the 'Standardized Code' column contains the correct code to use")
        print("- Set 'Keep Separate' to 'Yes' for participants that should not be merged")
        print("  (e.g., participants who filled out the same survey twice)")
        print(f"- Known duplicates to keep separate: participants 13, 15, 16, and 25")
        
        # Print the mapping for reference
        print("\nPotential duplicates found:")
        for norm_code, orig_codes in potential_duplicates.items():
            suggested = mapping_df[mapping_df['Normalized Code'] == norm_code]['Standardized Code'].iloc[0]
            print(f"  {norm_code}: {orig_codes} → {suggested}")
        
        # Step 3: Apply mapping (only if the mapping file exists and has been reviewed)
        if os.path.exists(mapping_file):
            user_input = input("\nHave you reviewed and edited the mapping file? (yes/no): ")
            if user_input.lower() == 'yes':
                corrected_df = apply_mapping(summary_file, mapping_file, corrected_file)
                print(f"Created corrected summary file: {corrected_file}")
            else:
                print("Please review and edit the mapping file, then run this script again.")
    else:
        print("No potential duplicates found.")

if __name__ == "__main__":
    main()
