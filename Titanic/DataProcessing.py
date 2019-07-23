import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv


def load_data_from_csv(file_name):
    training_data = pd.read_csv(file_name)
    return training_data


def plot_graph(data):
    sns.lmplot('Age', 'Sex', data, hue='Survived', fit_reg=False)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.show()


def drop_selected_columns(data, columns):
    return data.drop(columns, axis=1)


def generate_csv(predictions):
    cont = 892
    with open('results.csv', 'a') as csvFile:
        row = ['PassengerId', 'Survived']
        writer = csv.writer(csvFile)
        writer.writerow(row)
        for prediction in predictions:
            row = [str(cont), str(prediction)]
            writer = csv.writer(csvFile)
            writer.writerow(row)
            cont = cont + 1
    csvFile.close()
