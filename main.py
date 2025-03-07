import pandas as pd
import re  # Import regex module
from tabulate import tabulate  # Can be used for debugging

# Display settings for Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

print("starting the dnn file transformer")

# Function to rename columns based on prefixes
def rename_columns(df):
   
    
    new_columns = {}
    for col in df.columns:
        if re.match(r"^Q9_\d+", col):
            new_columns[col] = col.replace("Q9_", "PANAS_")
        elif re.match(r"^Q24_\d+", col):
            new_columns[col] = col.replace("Q24_", "BFI_")
        elif re.match(r"^Q25_\d+", col):
            new_columns[col] = col.replace("Q25_", "AReA_")
        elif re.match(r"^Q27_\d+", col):
            new_columns[col] = col.replace("Q27_", "Movement Familiarity_")


    renamed_df = df.rename(columns=new_columns)
    

    return renamed_df

# Read CSV files for Session 1
session_1_labels = pd.read_csv('data/session_1/labels.csv', skiprows=[1, 2])
session_1_values = pd.read_csv('data/session_1/values.csv', skiprows=[1, 2])

# Read CSV files for Session 2
session_2_labels = pd.read_csv('data/session_2/labels.csv', skiprows=[1, 2])
session_2_values = pd.read_csv('data/session_2/values.csv', skiprows=[1, 2])

# Ensure both dataframes contain "ResponseId"
if "ResponseId" not in session_1_labels.columns or "ResponseId" not in session_1_values.columns:
    print("Error: 'ResponseId' missing in Session 1 files")
    exit()

if "ResponseId" not in session_2_labels.columns or "ResponseId" not in session_2_values.columns:
    print("Error: 'ResponseId' missing in Session 2 files")
    exit()

# Apply renaming function

session_1_labels = rename_columns(session_1_labels)
session_1_values = rename_columns(session_1_values)


session_2_labels = rename_columns(session_2_labels)
session_2_values = rename_columns(session_2_values)


# Merge Session 1 labels and values
session1_combined = pd.merge(session_1_labels, session_1_values, on="ResponseId", how="inner")




# Remove columns that contain only null values
session1_combined_cleaned = session1_combined.dropna(axis=1, how="all")

# Preserve all non-survey categorical data
metadata_columns = ["ResponseId"]  # Start with ResponseId as essential metadata
non_survey_categorical_columns = [col for col in session1_combined_cleaned.columns
                                  if not any(col.startswith(prefix) for prefix in ["STAI Trait_", "STAI State_", "PANAS_", "BFI_", "AReA_", "Movement Familiarity_"])
                                  and session1_combined_cleaned[col].dtype == "object"]  # Ensures non-numeric survey labels are preserved


# Identify survey/assessment numeric columns
survey_prefixes = ["STAI Trait_", "STAI State_", "PANAS_", "BFI_", "AReA_", "Movement Familiarity_"]
survey_numeric_columns = [col for col in session1_combined_cleaned.columns
                          if any(col.startswith(prefix) for prefix in survey_prefixes)
                          and pd.api.types.is_numeric_dtype(session1_combined_cleaned[col])]

# # Keep metadata, categorical labels, and numeric responses only
# session1_combined_cleaned = session1_combined_cleaned[metadata_columns + non_survey_categorical_columns + survey_numeric_columns]


# Preserve original column order
original_column_order = session1_combined.columns

# Filter the correct columns
session1_combined_cleaned = session1_combined_cleaned[metadata_columns + non_survey_categorical_columns + survey_numeric_columns]

# Restore original order, keeping only selected columns
session1_combined_cleaned = session1_combined_cleaned[[col for col in original_column_order if col in session1_combined_cleaned.columns]]






# Rename again after merging (if needed)
session1_combined_cleaned = rename_columns(session1_combined_cleaned)

# Save the cleaned dataframe for Session 1
session1_output_filename = "session1_combined_with_null_columns_removed.csv"
session1_combined_cleaned.to_csv(session1_output_filename, index=False)
print(f"Session 1 file saved as {session1_output_filename}")

# Merge Session 2 labels and values
session2_combined = pd.merge(session_2_labels, session_2_values, on="ResponseId", how="inner")

# Remove columns that contain only null values
session2_combined_cleaned = session2_combined.dropna(axis=1, how="all")

# Preserve metadata (all non-survey categorical data)
metadata_columns = ["ResponseId"]  # Start with ResponseId as essential metadata
non_survey_categorical_columns = [col for col in session2_combined_cleaned.columns
                                  if not any(col.startswith(prefix) for prefix in ["STAI State_", "PANAS_"])
                                  and session2_combined_cleaned[col].dtype == "object"]

# Identify survey/assessment numeric columns
survey_prefixes = ["STAI State_", "PANAS_"]  # Only relevant surveys for Session 2
survey_numeric_columns = [col for col in session2_combined_cleaned.columns
                          if any(col.startswith(prefix) for prefix in survey_prefixes)
                          and pd.api.types.is_numeric_dtype(session2_combined_cleaned[col])]

# Keep metadata, categorical labels, and numeric responses only
session2_combined_cleaned = session2_combined_cleaned[metadata_columns + non_survey_categorical_columns + survey_numeric_columns]

# Ensure StartDate column exists before sorting
if "StartDate" in session2_combined_cleaned.columns:
    session2_combined_cleaned = session2_combined_cleaned.sort_values(by="StartDate")

# Ensure StartDate is the first column
if "StartDate" in session2_combined_cleaned.columns:
    cols = ["StartDate"] + [col for col in session2_combined_cleaned.columns if col != "StartDate"]
    session2_combined_cleaned = session2_combined_cleaned[cols]

# Overwrite the existing session2 CSV file instead of creating a new one
session2_combined_cleaned.to_csv("session2_combined_with_null_columns_removed.csv", index=False)




#At this point, the code outputs the columns with all the proper titles of the assessments and surveys. 
# The following task is to compute the total scores and sub-scores for: 

#State Trait Anxiety Inventory (STAI-S), measured during both sessions (scores of 20-80 possible) 

#(Reversed items are: 1,2, 5, 8, 10, 11,15, 16, 19, 20)

# Identify only numerical STAI-S columns dynamically for Session 1
stai_s_columns_s1 = [col for col in session1_combined_cleaned.columns if col.startswith("STAI State_")]

# Convert STAI-S columns to numeric (force non-numeric values to NaN) for Session 1
session1_combined_cleaned[stai_s_columns_s1] = session1_combined_cleaned[stai_s_columns_s1].apply(pd.to_numeric, errors='coerce')

# Identify reversed STAI-S items dynamically for Session 1
stai_s_reversed_s1 = [col for col in stai_s_columns_s1 if any(str(i) in col for i in [1, 2, 5, 8, 10, 11, 15, 16, 19, 20])]

# Apply reverse scoring for Session 1
if stai_s_reversed_s1:
    session1_combined_cleaned[stai_s_reversed_s1] = session1_combined_cleaned[stai_s_reversed_s1].replace({1: 4, 2: 3, 3: 2, 4: 1})

# Compute total STAI-S score for Session 1
if stai_s_columns_s1:
    session1_combined_cleaned["STAI-S Total"] = session1_combined_cleaned[stai_s_columns_s1].sum(axis=1, min_count=1)

# Insert STAI-S Total column **after the last STAI-S column** in Session 1
if stai_s_columns_s1 and "STAI-S Total" in session1_combined_cleaned.columns:
    last_stai_s_column_s1 = max([session1_combined_cleaned.columns.get_loc(col) for col in stai_s_columns_s1]) + 1
    session1_combined_cleaned.insert(last_stai_s_column_s1, "STAI-S Total", session1_combined_cleaned.pop("STAI-S Total"))


# --- Apply same logic for Session 2 ---

# Identify only numerical STAI-S columns dynamically for Session 2
stai_s_columns_s2 = [col for col in session2_combined_cleaned.columns if col.startswith("STAI State_")]

# Convert STAI-S columns to numeric (force non-numeric values to NaN) for Session 2
session2_combined_cleaned[stai_s_columns_s2] = session2_combined_cleaned[stai_s_columns_s2].apply(pd.to_numeric, errors='coerce')

# Identify reversed STAI-S items dynamically for Session 2
stai_s_reversed_s2 = [col for col in stai_s_columns_s2 if any(str(i) in col for i in [1, 2, 5, 8, 10, 11, 15, 16, 19, 20])]

# Apply reverse scoring for Session 2
if stai_s_reversed_s2:
    session2_combined_cleaned[stai_s_reversed_s2] = session2_combined_cleaned[stai_s_reversed_s2].replace({1: 4, 2: 3, 3: 2, 4: 1})

# Compute total STAI-S score for Session 2
if stai_s_columns_s2:
    session2_combined_cleaned["STAI-S Total"] = session2_combined_cleaned[stai_s_columns_s2].sum(axis=1, min_count=1)

# Insert STAI-S Total column **after the last STAI-S column** in Session 2
if stai_s_columns_s2 and "STAI-S Total" in session2_combined_cleaned.columns:
    last_stai_s_column_s2 = max([session2_combined_cleaned.columns.get_loc(col) for col in stai_s_columns_s2]) + 1
    session2_combined_cleaned.insert(last_stai_s_column_s2, "STAI-S Total", session2_combined_cleaned.pop("STAI-S Total"))


# Overwrite the existing session1 & session2 CSV files instead of creating new ones
session1_combined_cleaned.to_csv("session1_combined_with_null_columns_removed.csv", index=False)
session2_combined_cleaned.to_csv("session2_combined_with_null_columns_removed.csv", index=False)







#State Trait Anxiety Inventory (STAI-T), measured only during the train session (20-80 possible)
#(Reversed items are: 1,3, 6, 7, 10, 13,14, 16, 19)


# 1. Identify only numerical STAI-T columns dynamically
stai_t_columns = [col for col in session1_combined_cleaned.columns if col.startswith("STAI Trait_")]

# 2. Convert STAI-T columns to numeric (force non-numeric values to NaN)
session1_combined_cleaned[stai_t_columns] = session1_combined_cleaned[stai_t_columns].apply(pd.to_numeric, errors='coerce')

# 3. Identify reversed STAI-T items dynamically (filter only numeric ones)
stai_t_reversed = [col for col in stai_t_columns if any(str(i) in col for i in [1, 3, 6, 7, 10, 13, 14, 16, 19])]

# 4. Apply reverse scoring safely
if stai_t_reversed:
    session1_combined_cleaned[stai_t_reversed] = session1_combined_cleaned[stai_t_reversed].replace({1: 4, 2: 3, 3: 2, 4: 1})

# 5. Compute total STAI-T score (sum of all valid STAI-T items)
if stai_t_columns:
    session1_combined_cleaned["STAI-T Total"] = session1_combined_cleaned[stai_t_columns].sum(axis=1, min_count=1)

# 6. Dynamically find the last STAI-T column among numerical columns
if stai_t_columns and "STAI-T Total" in session1_combined_cleaned.columns:
    last_stai_column = max([session1_combined_cleaned.columns.get_loc(col) for col in stai_t_columns]) + 1
    session1_combined_cleaned.insert(last_stai_column, "STAI-T Total", session1_combined_cleaned.pop("STAI-T Total"))

# Ensure StartDate column exists before sorting
if "StartDate" in session1_combined_cleaned.columns:
    session1_combined_cleaned = session1_combined_cleaned.sort_values(by="StartDate")

# Ensure StartDate is the first column
if "StartDate" in session1_combined_cleaned.columns:
    cols = ["StartDate"] + [col for col in session1_combined_cleaned.columns if col != "StartDate"]
    session1_combined_cleaned = session1_combined_cleaned[cols]

# Overwrite the existing session1 CSV file instead of creating a new one
session1_combined_cleaned.to_csv("session1_combined_with_null_columns_removed.csv", index=False)







#PANAS (scores range from 10-50)
# For the total positive score, a higher score indicates more of a positive affect. 
# For the total negative score, a lower score indicates less ofa negative affect.

#Positive affect: questions 1, 3, 5, 9, 10, 12, 14, 16, 17, & 19
#Negative affect: questions 2, 4, 6, 7, 8, 11, 13, 15, 18, & 20 



# Identify all PANAS columns dynamically for Session 1
panas_columns_s1 = [col for col in session1_combined_cleaned.columns if col.startswith("PANAS_")]

# Convert PANAS columns to numeric (force non-numeric values to NaN) for Session 1
session1_combined_cleaned[panas_columns_s1] = session1_combined_cleaned[panas_columns_s1].apply(pd.to_numeric, errors='coerce')

# Define PANAS Positive & Negative Affect columns
panas_positive_s1 = [col for col in panas_columns_s1 if any(str(i) in col for i in [1, 3, 5, 9, 10, 12, 14, 16, 17, 19])]
panas_negative_s1 = [col for col in panas_columns_s1 if any(str(i) in col for i in [2, 4, 6, 7, 8, 11, 13, 15, 18, 20])]

# Compute total PANAS Positive score for Session 1
if panas_positive_s1:
    session1_combined_cleaned["PANAS Positive Total"] = session1_combined_cleaned[panas_positive_s1].sum(axis=1, min_count=1)

# Compute total PANAS Negative score for Session 1
if panas_negative_s1:
    session1_combined_cleaned["PANAS Negative Total"] = session1_combined_cleaned[panas_negative_s1].sum(axis=1, min_count=1)

# Insert PANAS Total columns **after the last PANAS column** in Session 1
if panas_columns_s1:
    last_panas_column_s1 = max([session1_combined_cleaned.columns.get_loc(col) for col in panas_columns_s1]) + 1
    if "PANAS Positive Total" in session1_combined_cleaned.columns:
        session1_combined_cleaned.insert(last_panas_column_s1, "PANAS Positive Total", session1_combined_cleaned.pop("PANAS Positive Total"))
    if "PANAS Negative Total" in session1_combined_cleaned.columns:
        session1_combined_cleaned.insert(last_panas_column_s1 + 1, "PANAS Negative Total", session1_combined_cleaned.pop("PANAS Negative Total"))

# Overwrite the existing session1 CSV file instead of creating a new one
session1_combined_cleaned.to_csv("session1_combined_with_null_columns_removed.csv", index=False)


#BFI, the extra short version

    #Open-Mindedness: 5, 10R, 15
        #Aesthetic Sensitivity: 5
        #Intellectual Curiosity: 10R
        #Creative Imagination: 15

    #Extraversion: 1R, 6, 11
        #Sociability: 1R 
        #Assertiveness: 6 
        #Energy Level: 11

    #Conscientiousness: 3R, 8R, 13
        #Organization: 3R
        #Productiveness: 8R
        #Responsibility: 13

    #Agreeableness: 2, 7R, 12
        #Compassion: 2
        #Respectfulness: 7R
        #Trust: 12

    #Negative Emotionality: 4, 9, 14R
        #Anxiety: 4
        #Depression: 9
        #Emotional Volatility: 14R


# Identify all BFI numerical columns dynamically (accounting for '_y' suffix)
bfi_columns = [col for col in session1_combined_cleaned.columns if col.startswith("BFI_")]

# Convert BFI columns to numeric (force non-numeric values to NaN)
session1_combined_cleaned[bfi_columns] = session1_combined_cleaned[bfi_columns].apply(pd.to_numeric, errors='coerce')

# Correctly identify reversed BFI items with the `_y` suffix
bfi_reversed = {
    "BFI_1_y": "Extraversion", "BFI_3_y": "Conscientiousness", "BFI_7_y": "Agreeableness",
    "BFI_8_y": "Conscientiousness", "BFI_10_y": "Open-Mindedness", "BFI_14_y": "Negative Emotionality"
}

# Ensure reversed BFI columns exist before applying reverse scoring
existing_bfi_reversed = [col for col in bfi_reversed.keys() if col in session1_combined_cleaned.columns]

# Apply reverse scoring safely
if existing_bfi_reversed:
    session1_combined_cleaned[existing_bfi_reversed] = session1_combined_cleaned[existing_bfi_reversed].replace({1: 5, 2: 4, 3: 3, 4: 2, 5: 1})

# Compute BFI subscores (now using `_y` format)
bfi_subscores = {
    "BFI_Open-Mindedness": ["BFI_5_y", "BFI_10_y", "BFI_15_y"],
    "BFI_Extraversion": ["BFI_1_y", "BFI_6_y", "BFI_11_y"],
    "BFI_Conscientiousness": ["BFI_3_y", "BFI_8_y", "BFI_13_y"],
    "BFI_Agreeableness": ["BFI_2_y", "BFI_7_y", "BFI_12_y"],
    "BFI_Negative Emotionality": ["BFI_4_y", "BFI_9_y", "BFI_14_y"]
}

# Ensure only existing columns are used in the computation
for subscore, cols in bfi_subscores.items():
    valid_cols = [col for col in cols if col in session1_combined_cleaned.columns]
    if valid_cols:
        session1_combined_cleaned[subscore] = session1_combined_cleaned[valid_cols].sum(axis=1, min_count=1)

# Find the last BFI question column dynamically
bfi_numeric_columns = [col for col in bfi_columns if col in session1_combined_cleaned.columns]
if bfi_numeric_columns:
    last_bfi_column_index = max([session1_combined_cleaned.columns.get_loc(col) for col in bfi_numeric_columns]) + 1
    for subscore in reversed(list(bfi_subscores.keys())):
        if subscore in session1_combined_cleaned.columns:
            session1_combined_cleaned.insert(last_bfi_column_index, subscore, session1_combined_cleaned.pop(subscore))

# Overwrite the existing session1 CSV file instead of creating a new one
session1_combined_cleaned.to_csv("session1_combined_with_null_columns_removed.csv", index=False)


#Aesthetic Responsiveness Assessment (AReA) (make sure all responses coded 1-5)
    #aesthetic appreciation (AA), Items 1, 2, 3, 4, 6, 9, 13, 14 (average)
    #intense aesthetic experience (IAE), Items 8, 11, 12, 13 (average)
    #creative behaviour, Items 5, 7, 10 (average)
    #overall AreA (Overall AReA scores are calculated as the average of all individual items)
        

# Identify AReA columns dynamically
area_columns = [col for col in session1_combined_cleaned.columns if col.startswith("AReA_")]

# Convert AReA columns to numeric (force non-numeric values to NaN)
session1_combined_cleaned[area_columns] = session1_combined_cleaned[area_columns].apply(pd.to_numeric, errors='coerce')

# Define subscore items
area_subscores = {
    "AReA_AA": [1, 2, 3, 4, 6, 9, 13, 14],
    "AReA_IAE": [8, 11, 12, 13],
    "AReA_CB": [5, 7, 10],
}

# Compute AReA subscores (averages)
for subscore, items in area_subscores.items():
    relevant_columns = [f"AReA_{i}_y" for i in items if f"AReA_{i}_y" in session1_combined_cleaned.columns]
    session1_combined_cleaned[subscore] = session1_combined_cleaned[relevant_columns].mean(axis=1, skipna=True)

# Compute Overall AReA score (average of all AReA items)
session1_combined_cleaned["AReA_Overall"] = session1_combined_cleaned[area_columns].mean(axis=1, skipna=True)

# Dynamically find last AReA column location
if area_columns and "AReA_Overall" in session1_combined_cleaned.columns:
    last_area_column = max([session1_combined_cleaned.columns.get_loc(col) for col in area_columns]) + 1
    for col in ["AReA_AA", "AReA_IAE", "AReA_CB", "AReA_Overall"]:
        session1_combined_cleaned.insert(last_area_column, col, session1_combined_cleaned.pop(col))

# Overwrite the existing session1 CSV file
session1_combined_cleaned.to_csv("session1_combined_with_null_columns_removed.csv", index=False)


#Fixes and exceptions (future section maybe input response id)





