from sklearn.ensemble import RandomForestClassifier


def generate_model(train_X, train_y):
    model = RandomForestClassifier(random_state=1)
    model.fit(train_X, train_y)
    return model
