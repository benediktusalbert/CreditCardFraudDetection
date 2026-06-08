import pandas as pd
import joblib
import time
import os

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

dataset = pd.read_csv("dataset/creditcard.csv")

print("\nDataset Loaded Successfully")
print(f"Total Rows: {len(dataset)}")

X = dataset.drop("Class", axis=1)
y = dataset["Class"]

print("\nScaling Data...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Scaling Finished")

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset Split Finished")

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=1
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        verbose=True
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=1,
        n_jobs=-1
    )
}

results = []

os.makedirs("models", exist_ok=True)

overall_start = time.time()

for index, (name, model) in enumerate(models.items(), start=1):

    print("\n" + "=" * 60)
    print(f"MODEL {index}/{len(models)}")
    print(f"Training: {name}")

    progress_bar = tqdm(
        total=100,
        desc=name,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}%"
    )

    start_time = time.time()

    progress_bar.update(10)

    model.fit(X_train, y_train)

    progress_bar.update(70)

    prediction = model.predict(X_test)

    progress_bar.update(20)

    end_time = time.time()

    progress_bar.close()

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    training_time = round(end_time - start_time, 2)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Training Time": training_time
    })

    filename = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(model, f"models/{filename}")

    print(f"\n{name} Finished")
    print(f"Training Time : {training_time} sec")
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")

overall_end = time.time()

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("ALL MODELS FINISHED")

print("\nModel Comparison:\n")
print(results_df)

results_df.to_csv("model_results.csv", index=False)

joblib.dump(scaler, "models/scaler.pkl")

total_time = round(overall_end - overall_start, 2)

print(f"\nTotal Training Time: {total_time} seconds")
print("All models saved successfully.")