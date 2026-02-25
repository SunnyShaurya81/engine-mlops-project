import sys, os, json, joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.data_ingestion import load_data
from components.model_trainer import train_model

def run_pipeline():

    train_df, test_df = load_data(
        "data/train_data.csv",
        "data/test_data.csv"
    )

    X_train = train_df.drop("Engine_Condition", axis=1)
    y_train = train_df["Engine_Condition"]

    X_test = test_df.drop("Engine_Condition", axis=1)
    y_test = test_df["Engine_Condition"]

    model, params = train_model(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    os.makedirs("model_registry", exist_ok=True)

    joblib.dump(model, "model_registry/engine_condition_model.joblib")

    with open("model_registry/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("✅ Training complete")
    print(metrics)

if __name__ == "__main__":
    run_pipeline()
