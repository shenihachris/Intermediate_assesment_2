# load data
import pandas as pd

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_file.csv")

# handle missing data
print(train_df.dtypes)
print(test_df.dtypes)

train_df.fillna(train_df.select_dtypes(include="number").mean(), inplace=True)
test_df.fillna(test_df.select_dtypes(include="number").mean(), inplace=True)

train_df.fillna(
    train_df.select_dtypes(include=["float64", "int64"]).mean(), inplace=True
)
test_df.fillna(test_df.select_dtypes(include=["float64", "int64"]).mean(), inplace=True)

# Encode catogorical variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
categorical_columns = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
]
for col in categorical_columns:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# feature engineering
train_df["Total_Income"] = train_df["ApplicantIncome"] + train_df["CoapplicantIncome"]
test_df["Total_Income"] = test_df["ApplicantIncome"] + test_df["CoapplicantIncome"]

# split training data
from sklearn.model_selection import train_test_split

X = train_df.drop(columns=["Loan_ID", "Loan_Status"])
y = train_df["Loan_Status"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# model training
print(X_train.head())
print(y_train.head())

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
X_train["Dependents"] = encoder.fit_transform(X_train["Dependents"])

X_train["Dependents"] = X_train["Dependents"].replace("3+", 3)

X_train = X_train.apply(pd.to_numeric, errors="coerce")
y_train = pd.to_numeric(y_train, errors="coerce")

print(X_train.isnull().sum())
print(y_train.isnull().sum())


if hasattr(model, "fit") and hasattr(model, "predict"):
    print("Model is properly trained!")
else:
    print("Model might not be trained!")

sample = test_df[:5]
predictions = model.predict(sample)
print(predictions)


# Make predictions
print(test_df.head())
print(test_df.columns)

print(train_df.dtypes)

numeric_columns = train_df.select_dtypes(include=["float64", "int64"]).columns
train_numeric = train_df[numeric_columns]

imputer = SimpleImputer(strategy="mean")
imputer.fit(train_numeric)
train_numeric_imputed = imputer.transform(train_numeric)

categorical_columns = train_df.select_dtypes(include=["object"]).columns
imputer_categorical = SimpleImputer(strategy="most_frequent")
train_categorical = train_df[categorical_columns]
train_categorical_imputed = imputer_categorical.fit_transform(train_categorical)

train_df_imputed = pd.concat(
    [
        pd.DataFrame(train_numeric_imputed, columns=numeric_columns),
        pd.DataFrame(train_categorical_imputed, columns=categorical_columns),
    ],
    axis=1,
)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy="mean"
)  # You can change this to "median" or "most_frequent"
X_train = imputer.fit_transform(X_train)


from sklearn.ensemble import HistGradientBoostingClassifier

model = HistGradientBoostingClassifier()
model.fit(X_train, y_train)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")  # Fill missing values
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Show predictions
print("Predictions:", y_pred)


train_df_features = train_df.drop(columns=["Loan_ID", "Loan_Status"])
train_df_labels = train_df["Loan_Status"]

import pandas as pd

train_df = pd.read_csv("train_data.csv")

print(train_df.columns)


print(train_df_features.columns)


print([col for col in train_df_features.columns])


train_df_features = train_df_features.ffill()


train_df_features = train_df_features.bfill()

print(train_df_features["Dependents"].unique())

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
train_df_features["Dependents"] = encoder.fit_transform(train_df_features["Dependents"])

train_df_features["Dependents"] = train_df_features["Dependents"].replace("3+", 3)
train_df_features["Dependents"] = pd.to_numeric(
    train_df_features["Dependents"], errors="coerce"
)
train_df_features = train_df_features.fillna(method="ffill")

print(train_df_features.columns)
print(train_df_features["Dependents"].unique())

print([col for col in train_df_features.columns])

# prepare submission file


import pandas as pd

# Example: Assuming 'data' is your original DataFrame
data["Loan_Status"] = "N/A"  # Add the column with default value (or populate as needed)

# Save the DataFrame to a new file
data.to_csv("submission_file.csv", index=False)

print("Submission file created successfully!")
