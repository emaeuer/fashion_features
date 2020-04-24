import csv
import matplotlib.pyplot as plt

# create data structure
# empty values represent unused columns
categories = ['', 'gender', 'masterCategory', 'subCategory', 'articleType', 
    'baseColour', '', '', 'usage', '']

distribution = {x:{} for x in categories}
distribution.pop("")

# read csv and insert data into data structure
with open("fashion-dataset/styles.csv") as file:
    reader = csv.reader(file)
    for row in reader:
        for category, value in zip(categories, row[0:]):
            if category is '':
                continue
            else:
                if value in distribution[category]:
                    distribution[category][value] += 1
                else:
                    distribution[category][value] = 1

# evaluate data structur and write result in text file and figures
with open("fashion_features/project/evaluation/results/result.txt", "w", encoding="UTF-8") as file:
    for category in distribution.items():
        
        category_name = category[0]
        distribution_of_category = dict(sorted(category[1].items(), key=lambda item: item[1], reverse=True))
       
        # write data of current category in the text file
        file.write("============ {0} ============\n".format(category_name))
        for attribute in distribution_of_category.items():
            file.write("{0}: {1}\n".format(attribute[0], attribute[1]))
        file.writelines("\n")

        # create and save figure for current category
        data = distribution_of_category.values()
        labels = distribution_of_category.keys()

        fig_width = 4 + len(labels)

        plt.figure(figsize=(fig_width, 10))
        plt.ylabel('Number of pictures')
        plt.title(category_name)
        plt.xticks(rotation=90)
        plt.bar(labels, data) 
        plt.savefig(f"fashion_features/project/evaluation/results/{category_name}.png")
    