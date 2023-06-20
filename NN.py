import keras
from keras.models import Sequential
from keras.layers import Dense
model = keras.Model

def init():
    global model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(23,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


def train(X_train, Y_train, X_val, Y_val):
    hist = model.fit(X_train, Y_train,
                     batch_size=16, epochs=30,
                     validation_data=(X_val, Y_val))
    print(model.summary())
    print(hist)
    return model

def eval(X_test, Y_test):
    return model.evaluate(X_test, Y_test)[1]