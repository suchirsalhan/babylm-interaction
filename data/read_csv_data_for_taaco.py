import csv
import pandas as pd

file_path = "../text_generation_evaluation/Newsroom.csv"  #SummEval.csv"

# Read csv to a DataFrame
df = pd.read_csv(file_path)

print("Columns", df.columns)
print("len of data:", len(df))

for index,row in df.iterrows():
    filename = row['id']
    src = row['src']
    text = row['sys']

    #with open("SummEval/generated/"+str(filename)+".txt", 'w') as f1:
    with open("Newsroom/generated/"+str(filename)+".txt", 'w') as f1:
        f1.write(text)

    #with open("SummEval/source/"+str(filename)+".txt", 'w') as f1:
    with open("Newsroom/source/"+str(filename)+".txt", 'w') as f1:
        f1.write(src)



