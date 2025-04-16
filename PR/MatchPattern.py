import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load Titanic dataset from Seaborn
df = sns.load_dataset('titanic')

# Drop rows with missing values in key columns
df = df[['pclass', 'sex', 'age', 'survived']].dropna()

# Encode 'sex' to numeric
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])  # male=1, female=0

# Feature selection menu
print("Available features to use: pclass, sex, age")
features_to_use = input("Enter features you want to use (comma-separated): ").lower().split(',')

# Clean user input
features_to_use = [f.strip() for f in features_to_use]

# Get user input for the selected features
user_input = {}
for feature in features_to_use:
    value = input(f"Enter value for {feature}: ")
    if feature == 'sex':
        value = le.transform([value.lower()])[0]
    else:
        value = float(value)
    user_input[feature] = [value]

# Prepare training data
X = df[features_to_use]
y = df['survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict using user input
user_df = pd.DataFrame(user_input)
prediction = model.predict(user_df)

print("\nPrediction Result:")
print("Survived" if prediction[0] == 1 else "Did not survive")
