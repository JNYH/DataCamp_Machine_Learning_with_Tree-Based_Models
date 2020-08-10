## CLASSIFICATION
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Import train_test_split
from sklearn.model_selection import train_test_split

# Import accuracy_score
from sklearn.metrics import accuracy_score

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y,test_size=0.2,stratify=y,random_state=1)

# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

# Instantiate dt, set 'criterion' to 'gini'
dt = DecisionTreeClassifier(criterion= 'gini', random_state=1)

# Instantiate dt, set 'entropy' as the information criterion
dt = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=1)

# Most of the time, the gini index and entropy lead to the same results.
# The gini index is slightly faster to compute and is the default criterion used in the DecisionTreeClassifier model of scikit-learn

# Fit dt to the training set
dt.fit(X_train,y_train)

# Predict test set labels
y_pred = dt.predict(X_test)

# Evaluate test-set accuracy
acc = accuracy_score(y_test, y_pred)
print("Test set accuracy: {:.2f}".format(acc))




## REGRESSION
# Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

# Import train_test_split
from sklearn.model_selection import train_test_split

# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE

# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=3)

# Instantiate a DecisionTreeRegressor 'dt'
# min_samples_leaf=0.1 implies at least 10% of the training data
dt = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1, random_state=3)

# Fit 'dt' to the training-set
dt.fit(X_train, y_train)

# Predict test-set labels
y_pred = dt.predict(X_test)

# Compute test-set MSE
mse_dt = MSE(y_test, y_pred)

# Compute test-set RMSE
rmse_dt = mse_dt**(1/2)

# Print rmse_dt
print(rmse_dt)
print("Test set RMSE of dt: {:.2f}".format(rmse_dt))
