# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# import CSV
lith = pd.read_csv(
    r"/Users/denise/Library/CloudStorage/OneDrive-Personal/CSU GLOBAL/MIS 581 Capstone/Raw Data/Lith_XRF_joined_Final.csv")
print(lith.head())

# create pandas dataframe and define column names
df = pd.DataFrame(lith, columns=['Al_pct', 'Si_pct', 'S_pct', 'K_pct', 'Ca_pct', 'Ti_ppm', 'Mn_ppm', 'Fe_pct', 'Cu_ppm',
                                 'Rb_ppm', 'Sr_ppm', 'Major_Unit_Interp'])

# Define predictor and target variables
X = df.drop('Major_Unit_Interp', axis=1)  # predictors
y = df['Major_Unit_Interp']  # target

#view first five rows of X and y
print(X.head())
print(y.head())

stats = df.describe(include='all').to_csv(r"/Users/denise/Library/CloudStorage/OneDrive-Personal/CSU GLOBAL/MIS 581 Capstone/Raw Data/Stats.csv")
print(stats)


# Split features and target into train (70%) and test sets (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test

# Create random forest with 500 trees
forest = RandomForestClassifier(n_estimators = 500)
forest.fit(X_train, y_train)

# Predict the test set using the forest
y_pred_test = forest.predict(X_test)

# View random forest accuracy score
print(accuracy_score(y_test, y_pred_test))

# View confusion matrix for test data and predictions
print(confusion_matrix(y_test, y_pred_test))

# Reshape the confusion matrix for plotting
matrix = confusion_matrix(y_test, y_pred_test)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build plot for confusion matrix
plt.figure(figsize=(14,6))
sns.set(font_scale=1.2)
sns.heatmap(matrix, annot=True, annot_kws={'size':8},
            cmap=plt.cm.Reds, linewidths=0.2)

# label and show the confusion matrix as heat map
class_names = ['Basal Conglomerate'
,'Basalt'
,'Diabase'
,'Diorite'
,'Gila Conglomerate'
,'Oracle Granite'
,'Porphyry']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()


# View the classification report for test data and predictions
print(classification_report(y_test, y_pred_test))

# Create feature importances for predictors
feature_scores = pd.Series(forest.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# View feature importance
print(feature_scores)


importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)

#plot feature importances
plt.figure(figsize=(14,6))
plt.bar(range(len(forest.feature_importances_)), forest.feature_importances_)
plt.xticks(range(len(forest.feature_importances_)), X_train.columns)
plt.xlabel("Feature")
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()
