import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

def read_mapping_file(filename):
    """Create a dictionary which contains the mappings of the file"""
    mappings = {}
    with open(filename) as mapping_file:
        for mapping in mapping_file:
            value, keys = mapping.strip().split(":")
            if value is "":
                continue
            for key in keys.split(","):
                mappings[key] = value
    return mappings

def evaluate_data(data_frame, category_name):
    """Create a plot for the column of a data frame"""
    series = data_frame[category_name]
    counts_of_groups = series.value_counts()
    counts_of_groups.plot.bar(y="count", rot=0, figsize=(10, 10))
    plt.savefig(f"project/evaluation/results/{category_name}.png")
    plt.close()

# read csv
with open("styles.csv") as file:
    df = pd.read_csv(file, usecols=[0, 1, 4, 5, 6, 8])

    df.gender = df.gender.map(read_mapping_file("project/evaluation/mappings/gender_mapping.txt"))
    df.baseColour = df.baseColour.map(read_mapping_file("project/evaluation/mappings/color_mapping.txt"))
    df.articleType = df.articleType.map(read_mapping_file("project/evaluation/mappings/article_mapping.txt"))

    # remove lines which contain empty values
    df = df.dropna()

    df.to_csv("styles_edited.csv", index=False)

    evaluate_data(df, "gender")
    evaluate_data(df, "baseColour")
    evaluate_data(df, "articleType")
    evaluate_data(df, "season")
    evaluate_data(df, "usage")
