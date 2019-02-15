import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score as cv_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import svm
import seaborn as sns

df0 = pd.read_csv('input/0.csv', header=None)
df1 = pd.read_csv('input/1.csv', header=None)
df2 = pd.read_csv('input/2.csv', header=None)
df3 = pd.read_csv('input/3.csv', header=None)
data = pd.concat([df0, df1, df2, df3])

X = data.drop(64, axis=1)
y = data[64]

# X.head()

def new_feat(deg):
    for i in range(64):
        X[i+64] = X[i] ** deg

new_feat(2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# %%time
rf_model = RandomForestClassifier(n_estimators=650, random_state=0, criterion='entropy', n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_y_pred = rf_model.predict(X_test)
print("Accuracy on test set: {}".format(np.mean(rf_y_pred == y_test)))
# Accuracy on test set: 0.9407534246575342

# %%time
rf_score = cv_score(estimator=rf_model, X=X, y=y, cv=5, n_jobs=-1)
print('cv mean score {}'.format(rf_score.mean()))
# cv mean score 0.9245700607148877

# %%time
# rf_model = RandomForestClassifier(random_state=0, n_jobs=-1, criterion='entropy')
# param_grid = {
#     'n_estimators': np.arange(500, 801, 100),
#     'max_depth' : np.arange(10, 21, 3),
# }
# clf = GridSearchCV(rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
# clf.fit(X_train, y_train)
# clf.best_params_
# clf.best_score_
# rf_y_pred = clf.best_estimator_.predict(X_test)
# np.mean(rf_y_pred == y_test)

# Попробовал также svm классификатор, в статье описано, что он лучше всего работает, но у меня качество немного хуже,
# чем у леса. Вообще еще думал, может catboost попробовать, но я его еще изучаю)