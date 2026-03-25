def predict_proba(model, X):

    return model.predict_proba(X)[:,1][0]