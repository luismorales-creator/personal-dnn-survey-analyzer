import pandas as pd
import json
from tabulate import tabulate
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Don't wrap wide displays
pd.set_option('display.max_rows', None)     # Show all rows

print("starting the dnn file transformer")

# import labels csv into python data structure
session_1_labels = pd.read_csv('data/session_1/labels.csv', skiprows = [1,2]) # <--- this is called a dataframe
session_1_values = pd.read_csv('data/session_1/values.csv', skiprows = [1,2]) # <--- this is called a dataframe

session_2_labels = pd.read_csv('data/session_2/labels.csv', skiprows = [1,2]) # <--- this is called a dataframe
session_2_values = pd.read_csv('data/session_2/values.csv', skiprows = [1,2]) # <--- this is called a dataframe

# print a preview of the dataframe
# print(session_1_labels.head(n=5))
# print("#######")
# print(session_1_values.head(n=5))

combined_df = pd.merge(session_1_labels, session_1_values, on="ResponseId", how="inner")
# filtered_columns = ["StartDate_x", "IPAddress_x", "ResponseId"]
# print("all the columns: ", combined_df.columns)
# print(combined_df[filtered_columns])
original_columns = combined_df.columns
combined_with_null_columns_removed_df = combined_df.dropna(axis=1, how="all")
cleaned_columns = combined_with_null_columns_removed_df.columns

# print("Columns with only nulls", set(original_columns) - set(cleaned_columns)) # <----this prints the null columns
output_filename = "combined_with_null_columns_removed_df.csv"
combined_with_null_columns_removed_df.to_csv(output_filename, index=False)

# print(tabulate(combined_df))
# combined_df.to_csv("combined.csv", index=False)



