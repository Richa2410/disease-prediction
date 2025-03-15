import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


data = pd.read_csv("C:/Users/iitia/Downloads/archive/data.csv")

X = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']

imputer = SimpleImputer(strategy='mean')  # Replace NaNs with mean
X = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)
y_pred = svc_model.predict(X_test_scaled)  # Predict labels for test data

comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')


