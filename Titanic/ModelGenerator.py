from sklearn.neighbors import KNeighborsClassifier


def generate_model(train_X, train_y):
    model = KNeighborsClassifier(5)
    model.fit(train_X, train_y)
    return model
