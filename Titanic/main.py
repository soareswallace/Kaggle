from DataProcessing import load_data_from_csv
from ModelGenerator import generate_model
from sklearn.metrics import mean_absolute_error
import csv
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
csv_train_file_name = 'train.csv'
csv_test_file_name = 'test.csv'


def main():
    training_data = load_data_from_csv(csv_train_file_name, True)
    test_data = load_data_from_csv(csv_test_file_name, True)

    X = training_data[features]
    y = training_data.Survived
    test_set = test_data[features]

    model = generate_model(X, y)
    predictions = model.predict(test_set)

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


if __name__ == '__main__':
    main()