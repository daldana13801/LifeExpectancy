import mlflow
import mlflow.sklearn
from preprocess import load_config, load_data, preprocess_data, split_data
from model import train_model, evaluate_model

def main():
    config = load_config()
    df = load_data(config["data_path"])
    X, y = preprocess_data(df, config["target_column"])
    X_train, X_test, y_train, y_test = split_data(X, y, config["test_size"], config["random_state"])

    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"]
        })

        model = train_model(X_train, y_train, config["n_estimators"], config["max_depth"])
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
