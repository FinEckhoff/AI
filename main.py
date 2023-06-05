# This is a sample Python script.
import keras

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import adult
import NN
import svm
def main():
    NN.init()
    model = NN.train(adult.X_train, adult.Y_train, adult.X_val, adult.Y_val)
    NN.eval(adult.X_test, adult.Y_test)
    model.save("./model.keras")
    """
    svm.init()
    model = svm.train(adult.X_train, adult.Y_train)
    eval = svm.eval(adult.X_test, adult.Y_test)
    print(eval)
    """




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
