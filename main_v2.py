import pandas as pd
import re
from pathlib import Path

# Display settings for Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

print("Starting the survey data transformer")

# Constants for survey scoring
STAI_STATE_REVERSED_ITEMS = [1, 2, 5, 8, 10, 11, 15, 16, 19, 20]
STAI_TRAIT_REVERSED_ITEMS = [1, 3, 6, 7, 10, 13, 14, 16, 19]
PANAS_POSITIVE_ITEMS = [1, 3, 5, 9, 10, 12, 14, 16, 17, 19]
PANAS_NEGATIVE_ITEMS = [2, 4, 6, 7, 8, 11, 13, 15, 18, 20]
BFI_REVERSED_ITEMS = {"BFI_1": "Extraversion", "BFI_3": "Conscientiousness", 
                      "BFI_7": "Agreeableness", "BFI_8": "Conscientiousness", 
                      "BFI_10": "Open-Mindedness", "BFI_14": "Negative Emotionality"}
BFI_SUBSCORES = {
    "BFI_Open-Mindedness": [5, 10, 15],
    "BFI_Extraversion": [1, 6, 11],
    "BFI_Conscientiousness": [3, 8, 13],
    "BFI_Agreeableness": [2, 7, 12],
    "BFI_Negative Emotionality": [4, 9, 14]
}
AREA_SUBSCORES = {
    "AReA_AA": [1, 2, 3, 4, 6, 9, 13, 14],
    "AReA_IAE": [8, 11, 12, 13],
    "AReA_CB": [5, 7, 10]
}

# Output files
OUTPUT_SESSION1 = "session1_combined_with_null_columns_removed.csv"
OUTPUT_SESSION2 = "session2_combined_with_null_columns_removed.csv"
OUTPUT_SUMMARY = "summary_file.csv"

def rename_columns(df, session):
    """Rename columns based on prefixes to standardize column names"""
    new_columns = {}
    for col in df.columns:
        # Only rename Q9_ columns in Session 2, but leave them untouched in Session 1
        if session == 2 and re.match(r"^Q9_\d+", col):
            new_columns[col] = col.replace("Q9_", "PANAS_")
        elif re.match(r"^Q24_\d+", col):
            new_columns[col] = col.replace("Q24_", "BFI_")
        elif re.match(r"^Q25_\d+", col):
            new_columns[col] = col.replace("Q25_", "AReA_")
        elif re.match(r"^Q27_\d+", col):
            new_columns[col] = col.replace("Q27_", "Movement Familiarity_")
        elif re.match(r"^Q26(\b|_)", col):
            new_columns[col] = col.replace("Q26", "Participation Code")
    
    return df.rename(columns=new_columns)

def load_and_preprocess_data(session_number):
    """Load and preprocess data for a session"""
    session_dir = f"data/session_{session_number}"
    labels_df = pd.read_csv(f"{session_dir}/labels.csv", skiprows=[1, 2])
    values_df = pd.read_csv(f"{session_dir}/values.csv", skiprows=[1, 2])
    
    # Rename columns
    labels_df = rename_columns(labels_df, session=session_number)
    values_df = rename_columns(values_df, session=session_number)
    
    # Merge labels and values
    combined_df = pd.merge(labels_df, values_df, on="ResponseId", how="inner")
    
    # Handle duplicate Participation Code columns
    participation_cols = [col for col in combined_df.columns if "Participation Code" in col]
    if len(participation_cols) > 1:
        combined_df.drop(columns=participation_cols[1:], inplace=True)
    
    # Standardize Participation Code column name
    if "Participation Code_x" in combined_df.columns:
        combined_df.rename(columns={"Participation Code_x": "Participation Code"}, inplace=True)
    
    # Clean column names
    combined_df.columns = combined_df.columns.str.strip()
    
    # Keep all columns, even those with all NaN values
    # combined_df = combined_df.dropna(axis=1, how="all")
    
    # Sort by StartDate if it exists
    if "StartDate" in combined_df.columns:
        combined_df = combined_df.sort_values(by="StartDate")
        # Move StartDate to first column
        cols = ["StartDate"] + [col for col in combined_df.columns if col != "StartDate"]
        combined_df = combined_df[cols]
    
    return combined_df

def convert_columns_to_numeric(df, prefix):
    """Convert columns with given prefix to numeric values"""
    cols = [col for col in df.columns if col.startswith(prefix)]
    if cols:
        df.loc[:, cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df, cols

def apply_reverse_scoring(df, columns, scale_type="5-point"):
    """Apply reverse scoring to specified columns"""
    if scale_type == "5-point":
        reverse_map = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
    elif scale_type == "4-point":
        reverse_map = {1: 4, 2: 3, 3: 2, 4: 1}
    else:
        raise ValueError(f"Unknown scale type: {scale_type}")
    
    if columns:
        df.loc[:, columns] = df[columns].replace(reverse_map)
    return df

def calculate_score(df, columns, operation="sum", name=None, min_count=1):
    """Calculate a score from a set of columns using sum or mean"""
    valid_columns = [col for col in columns if col in df.columns]
    
    if not valid_columns:
        return df, None
    
    if operation == "sum":
        result = df[valid_columns].sum(axis=1, min_count=min_count)
    elif operation == "mean":
        result = df[valid_columns].mean(axis=1, skipna=True)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    if name:
        # Create a Series with the score
        score_series = pd.Series(result, name=name)
        return df, score_series
    
    return df, result

def insert_score_after_columns(df, score_series, columns, name=None):
    """Insert a score column after the last column in a set"""
    if score_series is None or columns is None or not columns:
        return df
    
    # Find position to insert score
    last_column_index = max([df.columns.get_loc(col) for col in columns if col in df.columns]) + 1
    
    # Create DataFrame with the score
    score_df = pd.DataFrame({name if name else score_series.name: score_series})
    
    # Insert score at the correct position
    result_df = pd.concat(
        [df.iloc[:, :last_column_index],
         score_df,
         df.iloc[:, last_column_index:]],
        axis=1
    )
    
    return result_df

def process_stai_state(df, session_number):
    """Process STAI State data for a session"""
    # Convert columns to numeric
    df, stai_s_columns = convert_columns_to_numeric(df, "STAI State_")
    
    if not stai_s_columns:
        return df, None
    
    # Identify reversed items
    stai_s_reversed = [col for col in stai_s_columns 
                       if any(str(i) in col for i in STAI_STATE_REVERSED_ITEMS)]
    
    # Apply reverse scoring
    df = apply_reverse_scoring(df, stai_s_reversed, scale_type="4-point")
    
    # Calculate total score
    df, stai_s_total = calculate_score(
        df, stai_s_columns, operation="sum", 
        name=f"STAI-S Total_Session{session_number}"
    )
    
    # Insert total score after the last STAI-S column
    df = insert_score_after_columns(
        df, stai_s_total, stai_s_columns, 
        name=f"STAI-S Total_Session{session_number}"
    )
    
    return df, stai_s_columns

def process_stai_trait(df):
    """Process STAI Trait data"""
    # Convert columns to numeric
    df, stai_t_columns = convert_columns_to_numeric(df, "STAI Trait_")
    
    if not stai_t_columns:
        return df, None
    
    # Identify reversed items
    stai_t_reversed = [col for col in stai_t_columns 
                       if any(str(i) in col for i in STAI_TRAIT_REVERSED_ITEMS)]
    
    # Apply reverse scoring
    df = apply_reverse_scoring(df, stai_t_reversed, scale_type="4-point")
    
    # Calculate total score
    df, stai_t_total = calculate_score(
        df, stai_t_columns, operation="sum", 
        name="STAI-T Total"
    )
    
    # Insert total score after the last STAI-T column
    df = insert_score_after_columns(
        df, stai_t_total, stai_t_columns, 
        name="STAI-T Total"
    )
    
    return df, stai_t_columns

def process_panas(df, session_number):
    """Process PANAS data for a session"""
    # Check for PANAS columns with both formats
    panas_columns = [col for col in df.columns if col.startswith("PANAS_")]
    
    # Try alternate format if no columns found
    if not panas_columns:
        panas_space_columns = [col for col in df.columns if col.startswith("PANAS ")]
        if panas_space_columns:
            # Create a rename dictionary to convert "PANAS " to "PANAS_"
            rename_dict = {col: col.replace("PANAS ", "PANAS_") for col in panas_space_columns}
            df = df.rename(columns=rename_dict)
            # Update the list of PANAS columns after renaming
            panas_columns = [col for col in df.columns if col.startswith("PANAS_")]
    
    # Convert columns to numeric
    df, _ = convert_columns_to_numeric(df, "PANAS_")
    
    if not panas_columns:
        return df, None
    
    # Define PANAS Positive & Negative Affect columns
    panas_positive = [col for col in panas_columns 
                      if any(str(i) in col for i in PANAS_POSITIVE_ITEMS)]
    panas_negative = [col for col in panas_columns 
                      if any(str(i) in col for i in PANAS_NEGATIVE_ITEMS)]
    
    # Calculate positive score
    df, panas_positive_total = calculate_score(
        df, panas_positive, operation="sum", 
        name=f"PANAS Positive Total_Session{session_number}"
    )
    
    # Calculate negative score
    df, panas_negative_total = calculate_score(
        df, panas_negative, operation="sum", 
        name=f"PANAS Negative Total_Session{session_number}"
    )
    
    # Insert positive score after the last PANAS column
    df = insert_score_after_columns(
        df, panas_positive_total, panas_columns, 
        name=f"PANAS Positive Total_Session{session_number}"
    )
    
    # Find the new position after inserting the positive score
    last_panas_column_index = max([df.columns.get_loc(col) for col in panas_columns if col in df.columns]) + 1
    
    # Insert negative score after the positive score
    if panas_negative_total is not None:
        # Create DataFrame with the negative score
        neg_score_df = pd.DataFrame({f"PANAS Negative Total_Session{session_number}": panas_negative_total})
        
        # Insert after the positive score which is at last_panas_column_index + 1
        df = pd.concat(
            [df.iloc[:, :last_panas_column_index + 1],
             neg_score_df,
             df.iloc[:, last_panas_column_index + 1:]],
            axis=1
        )
    
    return df, panas_columns

def process_bfi(df):
    """Process BFI data"""
    # Convert columns to numeric
    df, bfi_columns = convert_columns_to_numeric(df, "BFI_")
    
    if not bfi_columns:
        return df, None
    
    # Identify reversed items with potential _y suffix
    bfi_reversed_cols = {}
    for col in bfi_columns:
        # Extract the number from column name
        match = re.search(r'BFI_(\d+)(_y)?', col)
        if match and int(match.group(1)) in [1, 3, 7, 8, 10, 14]:
            item_num = int(match.group(1))
            base_key = f"BFI_{item_num}"
            if base_key in BFI_REVERSED_ITEMS:
                bfi_reversed_cols[col] = BFI_REVERSED_ITEMS[base_key]
    
    # Apply reverse scoring
    existing_bfi_reversed = list(bfi_reversed_cols.keys())
    df = apply_reverse_scoring(df, existing_bfi_reversed, scale_type="5-point")
    
    # Compute BFI subscores
    subscore_results = {}
    for subscore, item_nums in BFI_SUBSCORES.items():
        # Find columns for this subscore
        cols = []
        for num in item_nums:
            # Match both BFI_num and BFI_num_y formats
            matching_cols = [col for col in bfi_columns if re.match(fr'BFI_{num}(_y)?$', col)]
            cols.extend(matching_cols)
        
        # Calculate subscore
        if cols:
            df, result = calculate_score(df, cols, operation="sum", name=subscore)
            subscore_results[subscore] = result
    
    # Find the last BFI column position
    if bfi_columns and subscore_results:
        last_bfi_column_index = max([df.columns.get_loc(col) for col in bfi_columns]) + 1
        
        # Create a DataFrame with all subscores
        bfi_subscores_df = pd.DataFrame(subscore_results)
        
        # Insert all BFI subscores at once
        df = pd.concat(
            [df.iloc[:, :last_bfi_column_index],
             bfi_subscores_df,
             df.iloc[:, last_bfi_column_index:]],
            axis=1
        )
    
    return df, bfi_columns

def process_area(df):
    """Process AReA data"""
    # Convert columns to numeric
    df, area_columns = convert_columns_to_numeric(df, "AReA_")
    
    if not area_columns:
        return df, None
    
    # Compute AReA subscores
    subscore_results = {}
    for subscore, item_nums in AREA_SUBSCORES.items():
        # Find columns for this subscore
        cols = []
        for num in item_nums:
            # Match both AReA_num and AReA_num_y formats
            matching_cols = [col for col in area_columns if re.match(fr'AReA_{num}(_y)?$', col)]
            cols.extend(matching_cols)
        
        # Calculate subscore
        if cols:
            df, result = calculate_score(df, cols, operation="mean", name=subscore)
            subscore_results[subscore] = result
    
    # Calculate overall AReA score
    df, overall_result = calculate_score(df, area_columns, operation="mean", name="AReA_Overall")
    if overall_result is not None:
        subscore_results["AReA_Overall"] = overall_result
    
    # Find the last AReA column position
    if area_columns and subscore_results:
        last_area_column_index = max([df.columns.get_loc(col) for col in area_columns]) + 1
        
        # Create a DataFrame with all subscores
        area_subscores_df = pd.DataFrame(subscore_results)
        
        # Insert all AReA subscores at once
        df = pd.concat(
            [df.iloc[:, :last_area_column_index],
             area_subscores_df,
             df.iloc[:, last_area_column_index:]],
            axis=1
        )
    
    return df, area_columns

def fix_participant_26_27_issue(session1_df):
    """Fix the issue where participant 27 was incorrectly labeled as 26 in session 1"""
    # Make a copy to avoid modifying the original dataframe
    df_fixed = session1_df.copy()
    
    # Extract participant numbers from participation codes
    def extract_participant_number(code):
        if not isinstance(code, str):
            return None
        match = re.search(r'[sS](\d+)', code)
        if match:
            return int(match.group(1))
        if code.lower() == 'so9bdb1':
            return 9
        return None
    
    df_fixed['Participant_Number'] = df_fixed['Participation Code'].apply(extract_participant_number)
    
    # Find the two entries for participant 26
    participant_26_entries = df_fixed[df_fixed['Participant_Number'] == 26]
    
    print(f"Found {len(participant_26_entries)} entries for participant 26 in session 1")
    
    # If there are two entries, the female one (Gender_y=2.0) should be participant 27
    if len(participant_26_entries) == 2:
        # Find the female participant 26 (the one that should be 27)
        female_26 = participant_26_entries[participant_26_entries['Gender_y'] == 2.0]
        
        if len(female_26) == 1:
            # Get the index of the female participant 26
            female_26_index = female_26.index[0]
            
            # Change the participation code to s27pdb1
            original_code = df_fixed.loc[female_26_index, 'Participation Code']
            df_fixed.loc[female_26_index, 'Participation Code'] = 's27pdb1'
            
            print(f"Renamed participant from {original_code} to s27pdb1 (female participant in session 1)")
        else:
            print(f"Warning: Expected 1 female participant 26, found {len(female_26)}")
    else:
        print(f"Warning: Expected 2 entries for participant 26, found {len(participant_26_entries)}")
    
    # Remove the temporary column
    df_fixed = df_fixed.drop(columns=['Participant_Number'])
    
    return df_fixed

def convert_gender_codes(df):
    """Convert gender codes from numeric to text labels"""
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

def create_summary_file(session1_df, session2_df):
    """Create a summary file with key metrics from both sessions"""
    # Define the columns to include in the summary file
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
    for df in [session1_df, session2_df]:
        for col in df.columns:
            if "Participation Code" in col:
                df.rename(columns={col: "Participation Code"}, inplace=True)
    
    # Merge sessions on Participation Code (outer join to keep all participants)
    summary_df = pd.merge(session1_df, session2_df, on="Participation Code", how="outer", suffixes=('', '_Session2'))
    
    # Drop '_y' suffix from columns 
    summary_df = summary_df.rename(columns=lambda x: x.rstrip("_y") if x.endswith("_y") else x)
    
    # Keep only the columns that exist in our summary_columns list
    existing_columns = [col for col in summary_columns if col in summary_df.columns]
    summary_df = summary_df[existing_columns]
    
    return summary_df

def process_data():
    """Main function to process all data"""
    # Create output directory if it doesn't exist
    Path("output").mkdir(exist_ok=True)
    
    # Process Session 1 data
    print("Processing Session 1 data...")
    session1_df = load_and_preprocess_data(1)
    
    # Fix the participant 26/27 issue in session 1
    print("Fixing participant 26/27 issue...")
    session1_df = fix_participant_26_27_issue(session1_df)
    
    # Process STAI-State for Session 1
    session1_df, _ = process_stai_state(session1_df, 1)
    
    # Process STAI-Trait (only in Session 1)
    session1_df, _ = process_stai_trait(session1_df)
    
    # Process PANAS for Session 1
    session1_df, _ = process_panas(session1_df, 1)
    
    # Process BFI (only in Session 1)
    session1_df, _ = process_bfi(session1_df)
    
    # Process AReA (only in Session 1)
    session1_df, _ = process_area(session1_df)
    
    # Save Session 1 results
    session1_df.to_csv(f"output/{OUTPUT_SESSION1}", index=False)
    print(f"Session 1 data saved to output/{OUTPUT_SESSION1}")
    
    # Process Session 2 data
    print("Processing Session 2 data...")
    session2_df = load_and_preprocess_data(2)
    
    # Process STAI-State for Session 2
    session2_df, _ = process_stai_state(session2_df, 2)
    
    # Process PANAS for Session 2
    session2_df, _ = process_panas(session2_df, 2)
    
    # Save Session 2 results
    session2_df.to_csv(f"output/{OUTPUT_SESSION2}", index=False)
    print(f"Session 2 data saved to output/{OUTPUT_SESSION2}")
    
    # Create summary file
    print("Creating summary file...")
    summary_df = create_summary_file(session1_df, session2_df)
    
    # Convert gender codes to text labels
    print("Converting gender codes to text labels...")
    summary_df = convert_gender_codes(summary_df)
    
    # Save the summary file
    summary_df.to_csv(f"output/{OUTPUT_SUMMARY}", index=False)
    print(f"Summary data saved to output/{OUTPUT_SUMMARY}")
    
    print("Data processing complete!")

if __name__ == "__main__":
    process_data()
