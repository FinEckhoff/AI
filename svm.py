from sklearn import svm, metrics

model = svm.SVC
def init():
    global  model
    model = svm.SVC(kernel='poly')  # Linear Kernel


def train(X_train, y_train):
    model.fit(X_train, y_train)
    return model

def eval(X_test, y_test):
    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    return acc

