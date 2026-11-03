import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('titanic.csv', sep=";")

print(df.describe(include='all'))

df = df.drop(columns=["Name", "Ticket Number", "Cabin", "Life Boat"])

if df['Survived'].dtype == 'object':
    df['Survived'] = df['Survived'].map({'Yes': 1, 'No': 0})

numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

non_numeric_columns = df.select_dtypes(include=['object']).columns
df[non_numeric_columns] = df[non_numeric_columns].fillna('missing')

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(df[non_numeric_columns])
encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder.get_feature_names_out(non_numeric_columns),
    index=df.index
)

df = pd.concat([encoded_df, df.drop(columns=non_numeric_columns)], axis=1)
print(df.head())

train_df, test_df = train_test_split(df, test_size=0.3, stratify=df['Survived'], random_state=42)
print(len(train_df), len(test_df))

X_train = train_df.drop(columns=['Survived'])
y_train = train_df['Survived']
X_test = test_df.drop(columns=['Survived'])
y_test = test_df['Survived']


def evaluate(model, X, y):
    y_pred = model.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return conf_matrix, accuracy, sensitivity, specificity


def draw_tree(model, filename=None):
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=X_train.columns, class_names=['Not Survived', 'Survived'], filled=True)
    if filename:
        plt.savefig(filename)
    plt.show()


dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
print(evaluate(dt_classifier, X_test, y_test))
draw_tree(dt_classifier, "overfitted_tree.pdf")

dt_pruned_alpha = DecisionTreeClassifier(random_state=42, ccp_alpha=0.005)
dt_pruned_alpha.fit(X_train, y_train)
print(evaluate(dt_pruned_alpha, X_test, y_test))
draw_tree(dt_pruned_alpha)

dt_pruned_leaves = DecisionTreeClassifier(random_state=42, max_leaf_nodes=3)
dt_pruned_leaves.fit(X_train, y_train)
print(evaluate(dt_pruned_leaves, X_test, y_test))
draw_tree(dt_pruned_leaves)