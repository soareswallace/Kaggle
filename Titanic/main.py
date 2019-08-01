from DataProcessing import load_data_from_csv, plot_graph, drop_selected_columns, generate_csv
from ModelGenerator import generate_model
features = ['Parch', 'Fare']
csv_train_file_name = 'train.csv'
csv_test_file_name = 'test.csv'
dropped_columns = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Pclass', 'Age', 'SibSp']


def main():
    training_data = load_data_from_csv(csv_train_file_name)

    test_data = load_data_from_csv(csv_test_file_name)
    # plot_graph(training_data)

    training_data = drop_selected_columns(training_data, dropped_columns)
    training_data = training_data.fillna(training_data.mean())

    test_data = drop_selected_columns(test_data, dropped_columns)
    test_data = test_data.fillna(test_data.mean())

    X = training_data[features]
    y = training_data.Survived
    model = generate_model(X, y)

    predictions = model.predict(test_data)
    generate_csv(predictions)


if __name__ == '__main__':
    main()