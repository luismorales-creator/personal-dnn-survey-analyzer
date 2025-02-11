import pandas as pd

print("starting the dnn file transformer")

# import labels csv into python data structure
session_1_labels = pd.read_csv('data/session_1/labels.csv', skiprows = [1,2]) # <--- this is called a dataframe
session_1_values = pd.read_csv('data/session_1/values.csv', skiprows = [1,2]) # <--- this is called a dataframe

# print a preview of the dataframe
print(session_1_labels)
print(session_1_values)

# might need?
# if __name__ == "__main__":
#     print("hi Luis")
