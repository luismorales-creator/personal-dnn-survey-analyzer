import pandas as pd
import re  # Import regex module
from tabulate import tabulate  # Can be used for debugging

# Display settings for Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

print("starting the dnn file transformer")

# # Function to rename columns based on prefixes
# def rename_columns(df):
   
    
#     new_columns = {}
#     for col in df.columns:
#         if re.match(r"^Q9_\d+", col):
#             new_columns[col] = col.replace("Q9_", "PANAS_")
#         elif re.match(r"^Q24_\d+", col):
#             new_columns[col] = col.replace("Q24_", "BFI_")
#         elif re.match(r"^Q25_\d+", col):
#             new_columns[col] = col.replace("Q25_", "AReA_")
#         elif re.match(r"^Q27_\d+", col):
#             new_columns[col] = col.replace("Q27_", "Movement Familiarity_")



#     renamed_df = df.rename(columns=new_columns)
    

#     return renamed_df


def rename_columns(df, session):
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
        elif re.match(r"^Q26(\b|_)", col):  # Fix: Matches Q26, Q26_x, Q26_y, etc.
            new_columns[col] = col.replace("Q26", "Participation Code")  # Keeps _x, _y, etc.

    return df.rename(columns=new_columns)







# Read CSV files for Session 1
session_1_labels = pd.read_csv('data/session_1/labels.csv', skiprows=[1, 2])
session_1_values = pd.read_csv('data/session_1/values.csv', skiprows=[1, 2])

# Read CSV files for Session 2
session_2_labels = pd.read_csv('data/session_2/labels.csv', skiprows=[1, 2])
session_2_values = pd.read_csv('data/session_2/values.csv', skiprows=[1, 2])





# # Ensure both dataframes contain "ResponseId"
# if "ResponseId" not in session_1_labels.columns or "ResponseId" not in session_1_values.columns:
#     print("Error: 'ResponseId' missing in Session 1 files")
#     exit()

# if "ResponseId" not in session_2_labels.columns or "ResponseId" not in session_2_values.columns:
#     print("Error: 'ResponseId' missing in Session 2 files")
#     exit()




# # Apply renaming function

# session_1_labels = rename_columns(session_1_labels)
# session_1_values = rename_columns(session_1_values)


# session_2_labels = rename_columns(session_2_labels)
# session_2_values = rename_columns(session_2_values)


session_1_labels = rename_columns(session_1_labels, session=1)  # Leave PANAS names as is
session_1_values = rename_columns(session_1_values, session=1)  # Leave PANAS names as is

session_2_labels = rename_columns(session_2_labels, session=2)  # Rename Q9_ to PANAS_
session_2_values = rename_columns(session_2_values, session=2)  # Rename Q9_ to PANAS_




# # Merge Session 1 labels and values
# session1_combined = pd.merge(session_1_labels, session_1_values, on="ResponseId", how="inner")


# Merge Session 1 labels and values
session1_combined = pd.merge(session_1_labels, session_1_values, on="ResponseId", how="inner")

# Merge Session 1 labels and values
session1_combined = pd.merge(session_1_labels, session_1_values, on="ResponseId", how="inner")

# Remove duplicate "Participation Code" columns if they exist
participation_cols_s1 = [col for col in session1_combined.columns if "Participation Code" in col]
if len(participation_cols_s1) > 1:
    session1_combined.drop(columns=participation_cols_s1[1:], inplace=True)  # Keep only the first instance
session1_combined.rename(columns={"Participation Code_x": "Participation Code", "Participation Code_y": "Participation Code"}, inplace=True)




# Keep demographic columns in session1_combined
demographic_columns = ["Age", "Education", "Gender"]  # These exist in values.csv
existing_demographics = [col for col in demographic_columns if col in session1_combined.columns]





######

# Ensure demographic columns exist before filtering
demographic_columns = ["Age", "Education", "Gender"]
existing_demographics = [col for col in demographic_columns if col in session1_combined.columns]

# Extract demographic data separately
session1_demographics = session1_combined[existing_demographics]  # Keep only existing ones




############

# Rename columns before any filtering to ensure PANAS is correctly labeled
session1_combined = rename_columns(session1_combined, session=1)

# Strip all column names of leading/trailing spaces
session1_combined.columns = session1_combined.columns.str.strip()

# # Only replace internal spaces for PANAS columns
# session1_combined.rename(columns=lambda x: x.replace("PANAS ", "PANAS_") if x.startswith("PANAS ") else x, inplace=True)

session1_combined = session1_combined.rename(columns={
    "Participation Code_x": "Participation Code",
    "Age": "Age",
    "Gender": "Gender", 
    "Education": "Education"
}).drop(columns=["Participation Code_y"], errors='ignore')






# Ensure PANAS columns are numeric before filtering
panas_columns_s1 = [col for col in session1_combined.columns if col.startswith("PANAS_")]
session1_combined[panas_columns_s1] = session1_combined[panas_columns_s1].apply(pd.to_numeric, errors='coerce')



# Remove columns that contain only null values
session1_combined_cleaned = session1_combined.dropna(axis=1, how="all")

#Define survey prefixes
survey_prefixes = ["STAI Trait_", "STAI State_", "PANAS_", "BFI_", "AReA_", "Movement Familiarity_"]


# Preserve all non-survey categorical data
# Separate demographics from metadata


metadata_columns = ["ResponseId"]  # Essential non-demographic metadata

demographic_columns = ["Age", "Education", "Gender"]  # Key demographics

non_survey_categorical_columns = [col for col in session1_combined_cleaned.columns
                                  if not any(col.startswith(prefix) for prefix in ["STAI Trait_", "STAI State_", "PANAS_", "BFI_", "AReA_", "Movement Familiarity_"])
                                  and session1_combined_cleaned[col].dtype == "object"]  # Ensures non-numeric survey labels are preserved





# # Identify survey/assessment numeric columns

# survey_numeric_columns = [col for col in session1_combined_cleaned.columns
#                           if any(col.startswith(prefix) for prefix in survey_prefixes)
#                           and pd.api.types.is_numeric_dtype(session1_combined_cleaned[col])]


# Ensure PANAS columns are numeric before filtering
panas_columns_s1 = [col for col in session1_combined_cleaned.columns if col.startswith("PANAS_")]
# session1_combined_cleaned[panas_columns_s1] = session1_combined_cleaned[panas_columns_s1].apply(pd.to_numeric, errors='coerce')
session1_combined_cleaned.loc[:, panas_columns_s1] = session1_combined_cleaned[panas_columns_s1].apply(pd.to_numeric, errors='coerce') #<--- trying .loc operation
#to fix settingwithcopywarning
#This warning occurs when modifying a slice of a DataFrame instead of the original. To fix it, explicitly use .loc
#.loc[:, col] ensures Pandas modifies the actual DataFrame rather than a view, avoiding the ambiguous behavior that triggers the warning

# Identify survey/assessment numeric columns (AFTER ensuring PANAS is numeric)
survey_numeric_columns = [col for col in session1_combined_cleaned.columns
                          if any(col.startswith(prefix) for prefix in survey_prefixes)
                          and pd.api.types.is_numeric_dtype(session1_combined_cleaned[col])]





# # Keep metadata, categorical labels, demographics, and numeric responses only
# # session1_combined_cleaned = session1_combined_cleaned[
# #     metadata_columns + demographic_columns + non_survey_categorical_columns + survey_numeric_columns
# ]


# # Keep demographics, metadata, categorical labels, and numeric responses
# session1_combined_cleaned = session1_combined_cleaned[
#     metadata_columns + non_survey_categorical_columns + survey_numeric_columns + existing_demographics
# ]


# KEEP EVERYTHING (minimal filtering)
session1_combined_cleaned = session1_combined[session1_combined.columns]





# Preserve original column order
original_column_order = session1_combined.columns

# # Filter the correct columns
# session1_combined_cleaned = session1_combined_cleaned[metadata_columns + non_survey_categorical_columns + survey_numeric_columns]


# Instead of filtering specific columns, keep everything
session1_combined_cleaned = session1_combined_cleaned.copy()

# Restore original order, keeping only selected columns
session1_combined_cleaned = session1_combined_cleaned[[col for col in original_column_order if col in session1_combined_cleaned.columns]]






# # Rename again after merging (if needed)
# session1_combined_cleaned = rename_columns(session1_combined_cleaned)

# Save the cleaned dataframe for Session 1
session1_output_filename = "session1_combined_with_null_columns_removed.csv"
session1_combined_cleaned.to_csv(session1_output_filename, index=False)


# Merge Session 2 labels and values
session2_combined = pd.merge(session_2_labels, session_2_values, on="ResponseId", how="inner")


# Merge Session 2 labels and values
session2_combined = pd.merge(session_2_labels, session_2_values, on="ResponseId", how="inner")

# Remove duplicate "Participation Code" columns if they exist
participation_cols_s2 = [col for col in session2_combined.columns if "Participation Code" in col]
if len(participation_cols_s2) > 1:
    session2_combined.drop(columns=participation_cols_s2[1:], inplace=True)  # Keep only the first instance
session2_combined.rename(columns={"Participation Code_x": "Participation Code", "Participation Code_y": "Participation Code"}, inplace=True)


#Adding session argument


session2_combined = rename_columns(session2_combined, session=2)



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
    session1_combined_cleaned["STAI-S Total_Session1"] = session1_combined_cleaned[stai_s_columns_s1].sum(axis=1, min_count=1)


# Insert STAI-S Total column **after the last STAI-S column** in Session 1
if stai_s_columns_s1 and "STAI-S Total_Session1" in session1_combined_cleaned.columns:
    last_stai_s_column_s1 = max([session1_combined_cleaned.columns.get_loc(col) for col in stai_s_columns_s1]) + 1
    session1_combined_cleaned.insert(last_stai_s_column_s1, "STAI-S Total_Session1", session1_combined_cleaned.pop("STAI-S Total_Session1"))


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
    session2_combined_cleaned["STAI-S Total_Session2"] = session2_combined_cleaned[stai_s_columns_s2].sum(axis=1, min_count=1)


# Insert STAI-S Total column **after the last STAI-S column** in Session 2
if stai_s_columns_s2 and "STAI-S Total_Session2" in session2_combined_cleaned.columns:
    last_stai_s_column_s2 = session2_combined_cleaned.columns.get_loc(stai_s_columns_s2[-1]) + 1
    session2_combined_cleaned.insert(last_stai_s_column_s2, "STAI-S Total_Session2", session2_combined_cleaned.pop("STAI-S Total_Session2"))

# Overwrite the existing session1 & session2 CSV files instead of creating new ones
session1_combined_cleaned.to_csv("session1_combined_with_null_columns_removed.csv", index=False)
session2_combined_cleaned.to_csv("session2_combined_with_null_columns_removed.csv", index=False)








#State Trait Anxiety Inventory (STAI-T), measured only during the train session (20-80 possible)
#(Reversed items are: 1,3, 6, 7, 10, 13,14, 16, 19)


# 1. Identify only numerical STAI-T columns dynamically
stai_t_columns = [col for col in session1_combined_cleaned.columns if col.startswith("STAI Trait_")]

# 2. Convert STAI-T columns to numeric (force non-numeric values to NaN)
session1_combined_cleaned.loc[:, stai_t_columns] = session1_combined_cleaned[stai_t_columns].apply(pd.to_numeric, errors='coerce')

# 3. Identify reversed STAI-T items dynamically (filter only numeric ones)
stai_t_reversed = [col for col in stai_t_columns if any(str(i) in col for i in [1, 3, 6, 7, 10, 13, 14, 16, 19])]

# 4. Apply reverse scoring safely
if stai_t_reversed:
    session1_combined_cleaned.loc[:, stai_t_reversed] = session1_combined_cleaned[stai_t_reversed].replace({1: 4, 2: 3, 3: 2, 4: 1})

# 5. Compute total STAI-T score (sum of all valid STAI-T items)
stai_t_total = None
if stai_t_columns:
    stai_t_total = session1_combined_cleaned[stai_t_columns].sum(axis=1, min_count=1)

# 6. Find the last STAI-T column position
last_stai_column = max([session1_combined_cleaned.columns.get_loc(col) for col in stai_t_columns]) + 1 if stai_t_columns else 0

# 7. Create a new DataFrame with the STAI-T Total column
if stai_t_total is not None:
    # Create a DataFrame with the total score
    stai_t_df = pd.DataFrame({"STAI-T Total": stai_t_total})
    
    # Insert the STAI-T Total column at the right position using concat
    session1_combined_cleaned = pd.concat(
        [session1_combined_cleaned.iloc[:, :last_stai_column],  # Columns before STAI-T Total
         stai_t_df,  # STAI-T Total column
         session1_combined_cleaned.iloc[:, last_stai_column:]],  # Columns after STAI-T Total
        axis=1
    )

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


# Check for PANAS columns in session 1 - handle both formats (with and without space)
panas_columns_s1 = [col for col in session1_combined_cleaned.columns if col.startswith("PANAS_")]

# If no columns with "PANAS_" format are found, check for "PANAS " format and rename them
if not panas_columns_s1:
    panas_space_columns = [col for col in session1_combined_cleaned.columns if col.startswith("PANAS ")]
    if panas_space_columns:
        # Create a rename dictionary to convert "PANAS " to "PANAS_"
        rename_dict = {col: col.replace("PANAS ", "PANAS_") for col in panas_space_columns}
        session1_combined_cleaned = session1_combined_cleaned.rename(columns=rename_dict)
        # Update the list of PANAS columns after renaming
        panas_columns_s1 = [col for col in session1_combined_cleaned.columns if col.startswith("PANAS_")]
        print(f"Renamed {len(panas_space_columns)} PANAS columns to consistent format")

# Convert PANAS columns to numeric (force non-numeric values to NaN) for session 1 
if panas_columns_s1:
    session1_combined_cleaned[panas_columns_s1] = session1_combined_cleaned[panas_columns_s1].apply(pd.to_numeric, errors='coerce')

# Define PANAS Positive & Negative Affect columns for Session 1
panas_positive_s1 = [col for col in panas_columns_s1 if any(str(i) in col for i in [1, 3, 5, 9, 10, 12, 14, 16, 17, 19])]
panas_negative_s1 = [col for col in panas_columns_s1 if any(str(i) in col for i in [2, 4, 6, 7, 8, 11, 13, 15, 18, 20])]



# Compute total PANAS Positive & Negative scores for Session 1
if panas_positive_s1:
    session1_combined_cleaned["PANAS Positive Total_Session1"] = session1_combined_cleaned[panas_positive_s1].sum(axis=1, min_count=1)
if panas_negative_s1:
    session1_combined_cleaned["PANAS Negative Total_Session1"] = session1_combined_cleaned[panas_negative_s1].sum(axis=1, min_count=1)


# Insert PANAS Total columns **after the last PANAS column** in Session 1
if panas_columns_s1:
    last_panas_column_s1 = max([session1_combined_cleaned.columns.get_loc(col) for col in panas_columns_s1]) + 1
    if "PANAS Positive Total_Session1" in session1_combined_cleaned.columns:
        session1_combined_cleaned.insert(last_panas_column_s1, "PANAS Positive Total_Session1", session1_combined_cleaned.pop("PANAS Positive Total_Session1"))
    if "PANAS Negative Total_Session1" in session1_combined_cleaned.columns:
        session1_combined_cleaned.insert(last_panas_column_s1 + 1, "PANAS Negative Total_Session1", session1_combined_cleaned.pop("PANAS Negative Total_Session1"))


# Overwrite the existing session1 CSV files 

session1_combined_cleaned.to_csv("session1_combined_with_null_columns_removed.csv", index=False)




# Identify all PANAS columns dynamically for session 2
panas_columns_s2 = [col for col in session2_combined_cleaned.columns if col.startswith("PANAS_")]

# Convert PANAS columns to numeric (force non-numeric values to NaN) for session 1 

session2_combined_cleaned[panas_columns_s2] = session2_combined_cleaned[panas_columns_s2].apply(pd.to_numeric, errors='coerce')

# Define PANAS Positive & Negative Affect columns for Session 2
panas_positive_s2 = [col for col in panas_columns_s2 if any(str(i) in col for i in [1, 3, 5, 9, 10, 12, 14, 16, 17, 19])]
panas_negative_s2 = [col for col in panas_columns_s2 if any(str(i) in col for i in [2, 4, 6, 7, 8, 11, 13, 15, 18, 20])]

# Compute total PANAS Positive & Negative scores for Session 2
if panas_positive_s2:
    session2_combined_cleaned["PANAS Positive Total_Session2"] = session2_combined_cleaned[panas_positive_s2].sum(axis=1, min_count=1)
if panas_negative_s2:
    session2_combined_cleaned["PANAS Negative Total_Session2"] = session2_combined_cleaned[panas_negative_s2].sum(axis=1, min_count=1)



# Insert PANAS Total columns **after the last PANAS column** in Session 2
if panas_columns_s2:
    last_panas_column_s2 = max([session2_combined_cleaned.columns.get_loc(col) for col in panas_columns_s2]) + 1
    if "PANAS Positive Total_Session2" in session2_combined_cleaned.columns:
        session2_combined_cleaned.insert(last_panas_column_s2, "PANAS Positive Total_Session2", session2_combined_cleaned.pop("PANAS Positive Total_Session2"))
    if "PANAS Negative Total_Session2" in session2_combined_cleaned.columns:
        session2_combined_cleaned.insert(last_panas_column_s2 + 1, "PANAS Negative Total_Session2", session2_combined_cleaned.pop("PANAS Negative Total_Session2"))




# Overwrite the existing session2 CSV files 

session2_combined_cleaned.to_csv("session2_combined_with_null_columns_removed.csv", index=False)





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

# Convert BFI columns to numeric (force non-numeric values to NaN) - Using .loc to avoid SettingWithCopyWarning
session1_combined_cleaned.loc[:, bfi_columns] = session1_combined_cleaned[bfi_columns].apply(pd.to_numeric, errors='coerce')

# Correctly identify reversed BFI items with the _y suffix
bfi_reversed = {
    "BFI_1_y": "Extraversion", "BFI_3_y": "Conscientiousness", "BFI_7_y": "Agreeableness",
    "BFI_8_y": "Conscientiousness", "BFI_10_y": "Open-Mindedness", "BFI_14_y": "Negative Emotionality"
}

# Ensure reversed BFI columns exist before applying reverse scoring
existing_bfi_reversed = [col for col in bfi_reversed.keys() if col in session1_combined_cleaned.columns]

# Apply reverse scoring safely - Using .loc to avoid SettingWithCopyWarning
if existing_bfi_reversed:
    session1_combined_cleaned.loc[:, existing_bfi_reversed] = session1_combined_cleaned[existing_bfi_reversed].replace({1: 5, 2: 4, 3: 3, 4: 2, 5: 1})

# Compute BFI subscores (now using _y format)
bfi_subscores = {
    "BFI_Open-Mindedness": ["BFI_5_y", "BFI_10_y", "BFI_15_y"],
    "BFI_Extraversion": ["BFI_1_y", "BFI_6_y", "BFI_11_y"],
    "BFI_Conscientiousness": ["BFI_3_y", "BFI_8_y", "BFI_13_y"],
    "BFI_Agreeableness": ["BFI_2_y", "BFI_7_y", "BFI_12_y"],
    "BFI_Negative Emotionality": ["BFI_4_y", "BFI_9_y", "BFI_14_y"]
}

# Calculate all BFI subscores at once
subscore_results = {}
for subscore, cols in bfi_subscores.items():
    valid_cols = [col for col in cols if col in session1_combined_cleaned.columns]
    if valid_cols:
        subscore_results[subscore] = session1_combined_cleaned[valid_cols].sum(axis=1, min_count=1)

# Find the last BFI question column dynamically
bfi_numeric_columns = [col for col in bfi_columns if col in session1_combined_cleaned.columns]
if bfi_numeric_columns and subscore_results:
    # Find position to insert subscores
    last_bfi_column_index = max([session1_combined_cleaned.columns.get_loc(col) for col in bfi_numeric_columns]) + 1
    
    # Create DataFrame with all subscores
    bfi_subscores_df = pd.DataFrame(subscore_results)
    
    # Insert all BFI subscores at once using concat (prevents fragmentation)
    session1_combined_cleaned = pd.concat(
        [session1_combined_cleaned.iloc[:, :last_bfi_column_index],  # Columns before BFI subscores
         bfi_subscores_df,  # BFI subscore columns
         session1_combined_cleaned.iloc[:, last_bfi_column_index:]],  # Columns after BFI subscores
        axis=1
    )

# Overwrite the existing session1 CSV file instead of creating a new one
session1_combined_cleaned.to_csv("session1_combined_with_null_columns_removed.csv", index=False)


#Aesthetic Responsiveness Assessment (AReA) (make sure all responses coded 1-5)
    #aesthetic appreciation (AA), Items 1, 2, 3, 4, 6, 9, 13, 14 (average)
    #intense aesthetic experience (IAE), Items 8, 11, 12, 13 (average)
    #creative behaviour, Items 5, 7, 10 (average)
    #overall AreA (Overall AReA scores are calculated as the average of all individual items)
        

# # Identify AReA columns dynamically
# area_columns = [col for col in session1_combined_cleaned.columns if col.startswith("AReA_")]

# # Convert AReA columns to numeric (force non-numeric values to NaN)
# session1_combined_cleaned[area_columns] = session1_combined_cleaned[area_columns].apply(pd.to_numeric, errors='coerce')

# # Define subscore items
# area_subscores = {
#     "AReA_AA": [1, 2, 3, 4, 6, 9, 13, 14],
#     "AReA_IAE": [8, 11, 12, 13],
#     "AReA_CB": [5, 7, 10],
# }

# # Compute AReA subscores (averages)
# for subscore, items in area_subscores.items():
#     relevant_columns = [f"AReA_{i}_y" for i in items if f"AReA_{i}_y" in session1_combined_cleaned.columns]
#     session1_combined_cleaned[subscore] = session1_combined_cleaned[relevant_columns].mean(axis=1, skipna=True)

# # Compute Overall AReA score (average of all AReA items)
# session1_combined_cleaned["AReA_Overall"] = session1_combined_cleaned[area_columns].mean(axis=1, skipna=True)

# # Dynamically find last AReA column location
# if area_columns and "AReA_Overall" in session1_combined_cleaned.columns:
#     last_area_column = max([session1_combined_cleaned.columns.get_loc(col) for col in area_columns]) + 1
#     for col in ["AReA_AA", "AReA_IAE", "AReA_CB", "AReA_Overall"]:
#         session1_combined_cleaned.insert(last_area_column, col, session1_combined_cleaned.pop(col))


#^^^ This old code worked but created a PerformanceWarning (Fragmentation) The new code below fixes this by
#  batch-inserting columns instead of .insert() multiple times.


# Ensure AReA columns exist before computing scores
area_columns = [col for col in session1_combined_cleaned.columns if col.startswith("AReA_")]

if area_columns:
    # Convert AReA columns to numeric (force non-numeric values to NaN) - Fixes `SettingWithCopyWarning`
    session1_combined_cleaned.loc[:, area_columns] = session1_combined_cleaned.loc[:, area_columns].apply(pd.to_numeric, errors='coerce')

    # Define subscore items dynamically
    area_subscores = {
        "AReA_AA": [col for col in area_columns if any(str(i) in col for i in [1, 2, 3, 4, 6, 9, 13, 14])],
        "AReA_IAE": [col for col in area_columns if any(str(i) in col for i in [8, 11, 12, 13])],
        "AReA_CB": [col for col in area_columns if any(str(i) in col for i in [5, 7, 10])]
    }

    # Compute subscores using mean
    subscore_results = {
        subscore: session1_combined_cleaned[cols].mean(axis=1, skipna=True) 
        for subscore, cols in area_subscores.items() if cols
    }

    # Compute Overall AReA score (average of all AReA items)
    subscore_results["AReA_Overall"] = session1_combined_cleaned[area_columns].mean(axis=1, skipna=True)

    # # Use `pd.concat()` to add all computed scores at once - Fixes `PerformanceWarning`
    # session1_combined_cleaned = pd.concat([session1_combined_cleaned, pd.DataFrame(subscore_results)], axis=1) <---that code put the AReA subscores at the end of the csv, not directly after the last AReA question column
    

    #vvv this code will put the AReA subscores and total scores in the right place


# Find the last AReA column index dynamically
last_area_column_index = max([session1_combined_cleaned.columns.get_loc(col) for col in area_columns]) + 1 if area_columns else len(session1_combined_cleaned.columns)

# Create a DataFrame with computed subscores
area_subscores_df = pd.DataFrame(subscore_results)

# Insert the AReA subscores and overall score **right after the last AReA column**
session1_combined_cleaned = pd.concat(
    [session1_combined_cleaned.iloc[:, :last_area_column_index],  # Keep columns before AReA subscores
     area_subscores_df,  # Insert computed AReA scores
     session1_combined_cleaned.iloc[:, last_area_column_index:]  # Append the rest
    ], axis=1
)


# Overwrite the existing session1 CSV file
session1_combined_cleaned.to_csv("session1_combined_with_null_columns_removed.csv", index=False)






##### SUMMARY FILE #####



# Load the cleaned session files
session1_df = pd.read_csv("session1_combined_with_null_columns_removed.csv")
session2_df = pd.read_csv("session2_combined_with_null_columns_removed.csv")

# Ensure Participation Code is merged correctly
session1_df.rename(columns={"Participation Code_x": "Participation Code", "Participation Code_y": "Participation Code"}, inplace=True)
session2_df.rename(columns={"Participation Code_x": "Participation Code", "Participation Code_y": "Participation Code"}, inplace=True)

# Define final summary columns (dropping `_y` suffix for clarity)
summary_columns = [
    "Participation Code",
    "Gender", "Age", "Education",  # Keep demographic variables clean
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

# Merge sessions on Participation Code (outer join to keep all participants)
summary_df = pd.merge(session1_df, session2_df, on="Participation Code", how="outer")

# Drop `_y` suffix from columns like Gender, Education, Age, and Movement Familiarity
summary_df = summary_df.rename(columns=lambda x: x.rstrip("_y") if x.endswith("_y") else x)

# Keep only the desired summary columns
summary_df = summary_df[[col for col in summary_columns if col in summary_df.columns]]

# Save final summary file
summary_df.to_csv("summary_file.csv", index=False)



#Fixes and exceptions (future section maybe input response id)




#cd /Users/luisemilio/Documents/Code/personal_dnn_surveys


# participants missing session 2 surveys: 13, 15, 16, 25