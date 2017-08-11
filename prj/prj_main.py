from preprocess import transform
from preprocess import fill_missing
from preprocess import test_transform
from preprocess import test_fill_missing
from lr import LogitReg
from naive_bayes import NaiveBayes
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import sklearn.naive_bayes as NB
from sklearn.ensemble import RandomForestClassifier


def main():
    # load training data
    filename_test = './data/test.csv'
    X_test = test_transform(filename_test)


    filename_train = './data/train.csv'
    train_dataset = transform(filename_train)

    X = train_dataset['data']
    y = train_dataset['target']

    # fill in missing data (optional)
    X_full = fill_missing(X, 'most_frequent', False)

    X_train = X_full[:, 1:]

    User_ID = X_full[:, 0]

    User_ID = User_ID.reshape(User_ID.shape[0])



    # fill in missing data (optional)
    X_test = test_fill_missing(X_test, 'most_frequent', False)

    X_test = X_test[:, 1:]






    # Preprocessing over, data training starts
    ### use the logistic regression
    print('Train the logistic regression classifier')
    """ your code here """
    # LogisticRegression Model from sklearn
    lr_model = LogisticRegression()
    score = cross_val_predict(lr_model, X_train, y, cv=10)
    print("Sklearn Accuracy: %0.5f (+/- %0.5f)" % (score.mean(), score.std()))
    lr_model.fit(X_train, y)
    y_predict = lr_model.predict(X_test)

    fp = open("./predictions/lr_predictions.csv", "w")
    fp.write("UserID,Happy\n")
    for i in range(y_predict.shape[0]):
        line = "%d,%d\n" % (User_ID[i].astype(int), y_predict[i].astype(int))
        fp.write(line)
    fp.close()

    # LogisticRegression Model implemented by ourselves
    lr_model = LogitReg()
    score = cross_val_predict(lr_model, X_train, y, cv=10)
    print("Self-implemented Accuracy: %0.5f (+/- %0.5f)" % (score.mean(), score.std()))

    ### use the naive bayes
    print('Train the naive bayes classifier')
    """ your code here """
    # BernoulliNB Model from sklearn
    nb_model = NB.MultinomialNB()
    score = cross_val_predict(nb_model, X_train, y, cv=10)
    print("Sklearn Accuracy: %0.5f (+/- %0.5f)" % (score.mean(), score.std()))

    nb_model.fit(X_train, y)
    y_predict = nb_model.predict(X_test)
    fp = open("./predictions/nb_predictions.csv", "w")
    fp.write("UserID,Happy\n")
    for i in range(y_predict.shape[0]):
        line = "%d,%d\n" % (User_ID[i].astype(int), y_predict[i].astype(int))
        fp.write(line)
    fp.close()

    # NaiveBayes Model implemented by ourselves
    nb_model = NaiveBayes()
    score = cross_val_predict(nb_model, X_train, y, cv=10)
    print("Self-implemented Accuracy: %0.5f (+/- %0.5f)" % (score.mean(), score.std()))

    ## use the svm
    print('Train the SVM classifier')
    """ your code here """
    # Kernel function selection: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    kernel_fun = "rbf"
    svm_model = svm.SVC(kernel=kernel_fun)
    score = cross_val_predict(svm_model, X_train, y, cv=10)
    print("Accuracy: %0.5f (+/- %0.5f)" % (score.mean(), score.std()))

    svm_model.fit(X_train, y)
    y_predict = svm_model.predict(X_test)
    fp = open("./predictions/svm_predictions.csv", "w")
    fp.write("UserID,Happy\n")
    for i in range(y_predict.shape[0]):
        line = "%d,%d\n" % (User_ID[i].astype(int), y_predict[i].astype(int))
        fp.write(line)
    fp.close()

    ## use the random forest
    print('Train the random forest classifier')
    """ your code here """
    rf_model = RandomForestClassifier(n_jobs = -1,random_state =50,max_features = "auto",
                                      min_samples_leaf = 50)
    score = cross_val_predict(rf_model, X_train, y, cv=10)
    print("Accuracy: %0.5f (+/- %0.5f)" % (score.mean(), score.std()))

    rf_model.fit(X_train, y)
    y_predict = rf_model.predict(X_test)
    fp = open("./predictions/rf_predictions.csv", "w")
    fp.write("UserID,Happy\n")
    for i in range(y_predict.shape[0]):
        line = "%d,%d\n" % (User_ID[i].astype(int), y_predict[i].astype(int))
        fp.write(line)
    fp.close()

    ## get predictions
    """ your code here """
    # The prediction has been implemented above


if __name__ == '__main__':
    main()
