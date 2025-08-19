import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ===========================
# Feature engineering
# ===========================
def create_features(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mr':'Mr','Miss':'Miss','Mrs':'Mrs','Master':'Master',
        'Dr':'Rare','Rev':'Rare','Col':'Rare','Major':'Rare','Mlle':'Miss',
        'Countess':'Rare','Ms':'Mrs','Lady':'Rare','Jonkheer':'Rare',
        'Don':'Rare','Dona':'Rare','Mme':'Mrs','Capt':'Rare','Sir':'Rare'
    }
    df['Title'] = df['Title'].map(title_map).fillna('Rare')
    df['Deck'] = df['Cabin'].str[0].fillna('U')
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['IsChild'] = (df['Age'] < 16).astype(int)
    df['IsAdult'] = ((df['Age'] >= 16) & (df['Age'] < 60)).astype(int)
    df['IsElderly'] = (df['Age'] >= 60).astype(int)
    df['NameLength'] = df['Name'].str.len()
    return df

# ===========================
# Preprocessing
# ===========================
def preprocess(train_df, test_df):
    train_size = len(train_df)
    train_ids = train_df['PassengerId']
    test_ids = test_df['PassengerId']

    # Combine for processing
    test_df_copy = test_df.copy()
    test_df_copy['Survived'] = -1
    full_df = pd.concat([train_df, test_df_copy], ignore_index=True)

    # ===========================
    # Impute Age and Fare
    # ===========================
    full_df['Age'] = full_df.groupby(['Sex','Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    full_df['Age'] = full_df['Age'].fillna(full_df['Age'].median())

    full_df['Fare'] = full_df.groupby(['Pclass','Embarked'])['Fare'].transform(lambda x: x.fillna(x.median()))
    full_df['Fare'] = full_df['Fare'].fillna(full_df['Fare'].median())

    # Embarked
    full_df['Embarked'] = full_df['Embarked'].fillna(full_df['Embarked'].mode()[0])

    # Features
    full_df = create_features(full_df)

    # One-hot encoding for categorical
    categorical_cols = ['Sex','Embarked','Title','Deck']
    full_df = pd.get_dummies(full_df, columns=categorical_cols, drop_first=True)

    # Drop unused columns
    drop_cols = ['Name','Ticket','Cabin']
    full_df.drop(columns=[c for c in drop_cols if c in full_df.columns], inplace=True)

    # Split back
    X = full_df.iloc[:train_size].drop(columns=['PassengerId','Survived'])
    y = train_df['Survived']
    X_test = full_df.iloc[train_size:].drop(columns=['PassengerId','Survived'])

    # Align columns
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X.columns]

    # Scale numerical features
    num_cols = X.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Imputer for any remaining NaN
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    return X, y, X_test, test_ids

# ===========================
# Model training
# ===========================
def train_models(X_train, y_train, X_valid, y_valid):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric='logloss', random_state=42)
    }
    best_model = None
    best_score = 0
    trained_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        val_acc = accuracy_score(y_valid, y_pred)
        trained_models[name] = model
        print(f"{name}: CV={scores.mean():.4f}, Validation={val_acc:.4f}")
        if val_acc > best_score:
            best_score = val_acc
            best_model = model

    # Ensemble
    ensemble = VotingClassifier(
        estimators=[(n, m) for n, m in trained_models.items()],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_valid)
    ensemble_acc = accuracy_score(y_valid, ensemble_pred)
    print(f"Ensemble Validation Accuracy: {ensemble_acc:.4f}")
    if ensemble_acc > best_score:
        best_model = ensemble

    return best_model

# ===========================
# Main
# ===========================
def main():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    X, y, X_test, test_ids = preprocess(train_df, test_df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model = train_models(X_train, y_train, X_valid, y_valid)

    predictions = best_model.predict(X_test)

    submission = pd.DataFrame({"PassengerId": test_ids, "Survived": predictions})
    submission.to_csv("submission.csv", index=False)
    print("Submission created: submission.csv")

if __name__ == "__main__":
    main()
