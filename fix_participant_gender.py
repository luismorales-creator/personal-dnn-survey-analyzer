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

def fix_participant_26_27_issue(session1_df, session2_df):
    """Fix the issue where participant 27 was incorrectly labeled as 26 in session 1."""
    # Make copies to avoid modifying the original dataframes
    session1_fixed = session1_df.copy()
    session2_fixed = session2_df.copy()
    
    # Find the two entries for participant 26 in session 1
    session1_fixed['Participant_Number'] = session1_fixed['Participation Code'].apply(extract_participant_number)
    participant_26_entries = session1_fixed[session1_fixed['Participant_Number'] == 26]
    
    print(f"Found {len(participant_26_entries)} entries for participant 26 in session 1")
    
    # If there are two entries, the female one (Gender=2.0) should be participant 27
    if len(participant_26_entries) == 2:
        # Find the female participant 26 (the one that should be 27)
        female_26 = participant_26_entries[participant_26_entries['Gender_y'] == 2.0]
        
        if len(female_26) == 1:
            # Get the index of the female participant 26
            female_26_index = female_26.index[0]
            
            # Change the participation code to s27pdb1
            original_code = session1_fixed.loc[female_26_index, 'Participation Code']
            session1_fixed.loc[female_26_index, 'Participation Code'] = 's27pdb1'
            
            print(f"Renamed participant from {original_code} to s27pdb1 (female participant in session 1)")
        else:
            print(f"Warning: Expected 1 female participant 26, found {len(female_26)}")
    else:
        print(f"Warning: Expected 2 entries for participant 26, found {len(participant_26_entries)}")
    
    # Remove the temporary column
    session1_fixed = session1_fixed.drop(columns=['Participant_Number'])
    
    return session1_fixed, session2_fixed

def convert_gender_codes(df):
    """Convert gender codes from 1.0/2.0 to 'male'/'female'."""
    # Make a copy to avoid modifying the original dataframe
    df_fixed = df.copy()
    
    # Create a mapping for gender codes
    gender_mapping = {
        1.0: 'male',
        2.0: 'female',
        3.0: 'non-binary'  # In case there are other gender codes
    }
    
    # Convert gender codes
    if 'Gender' in df_fixed.columns:
        df_fixed['Gender'] = df_fixed['Gender'].map(lambda x: gender_mapping.get(x, x))
        print("Converted gender codes to text labels")
    
    return df_fixed

def main():
    # File paths
    session1_file = "output/session1_combined_with_null_columns_removed.csv"
    session2_file = "output/session2_combined_with_null_columns_removed.csv"
    summary_file = "output/summary_file_fixed.csv"
    output_file = "output/summary_file_gender_fixed.csv"
    
    # Read the data files
    print(f"Reading session 1 data from {session1_file}")
    session1_df = pd.read_csv(session1_file)
    
    print(f"Reading session 2 data from {session2_file}")
    session2_df = pd.read_csv(session2_file)
    
    print(f"Reading summary file from {summary_file}")
    summary_df = pd.read_csv(summary_file)
    
    # Fix the participant 26/27 issue
    print("\nFixing participant 26/27 issue...")
    session1_fixed, session2_fixed = fix_participant_26_27_issue(session1_df, session2_df)
    
    # Create a new summary file using the fixed session data
    print("\nCreating new summary file...")
    
    # Define the columns to include in the summary file (same as in main.py)
    summary_columns = [
        "Participation Code",
        "Gender", "Age", "Education",
        "STAI-T Total",
        "STAI-S Total_Session1", "STAI-S Total_Session2",
        "PANAS Positive Total_Session1", "PANAS Positive Total_Session2",
        "PANAS Negative Total_Session1", "PANAS Negative Total_Session2",
        "BFI_Open-Mindedness", "BFI_Extraversion", "BFI_Conscientiousness",
        "BFI_Agreeableness", "BFI_Negative Emotionality",
        "AReA_AA", "AReA_IAE", "AReA_CB", "AReA_Overall",
        "Movement Familiarity_1", "Movement Familiarity_2", "Movement Familiarity_3",
        "Movement Familiarity_4", "Movement Familiarity_5"
    ]
    
    # Standardize the Participation Code column name before merging
    for df in [session1_fixed, session2_fixed]:
        for col in df.columns:
            if "Participation Code" in col:
                df.rename(columns={col: "Participation Code"}, inplace=True)
    
    # Merge sessions on Participation Code (outer join to keep all participants)
    new_summary_df = pd.merge(session1_fixed, session2_fixed, on="Participation Code", how="outer", suffixes=('', '_Session2'))
    
    # Drop '_y' suffix from columns 
    new_summary_df = new_summary_df.rename(columns=lambda x: x.rstrip("_y") if x.endswith("_y") else x)
    
    # Keep only the columns that exist in our summary_columns list
    existing_columns = [col for col in summary_columns if col in new_summary_df.columns]
    new_summary_df = new_summary_df[existing_columns]
    
    # Convert gender codes to text labels
    new_summary_df = convert_gender_codes(new_summary_df)
    
    # Sort by participant number for better readability
    new_summary_df['Participant_Number'] = new_summary_df['Participation Code'].apply(extract_participant_number)
    new_summary_df = new_summary_df.sort_values('Participant_Number')
    new_summary_df = new_summary_df.drop(columns=['Participant_Number'])
    
    # Save the fixed summary file
    new_summary_df.to_csv(output_file, index=False)
    print(f"\nFixed summary file saved to {output_file}")
    
    # Print a comparison of the old and new summary files for participants 26 and 27
    print("\nComparison of participant 26 and 27 data:")
    print("\nOld summary file:")
    print(summary_df[summary_df['Participation Code'].isin(['s26pdb1', 's27pdb1'])])
    
    print("\nNew summary file:")
    print(new_summary_df[new_summary_df['Participation Code'].isin(['s26pdb1', 's27pdb1'])])

if __name__ == "__main__":
    main()
