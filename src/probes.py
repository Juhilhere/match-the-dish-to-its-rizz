from sklearn.linear_model import LogisticRegression, RidgeClassifier

def fit_logistic_probe(train_features, train_labels, c=2.5, max_iter=800, seed=42):
    clf = LogisticRegression(C=c, max_iter=max_iter, solver='lbfgs', multi_class='multinomial', random_state=seed)
    clf.fit(train_features, train_labels)
    return clf

def fit_ridge_probe(train_features, train_labels, alpha=1.5, seed=42):
    ridge = RidgeClassifier(alpha=alpha, random_state=seed)
    ridge.fit(train_features, train_labels)
    return ridge
