import mlflow
import mlflow.sklearn
#Funciones del proyecto
#Función para cargar los datos
from preprocess import load_config, load_data, preprocess_data, split_data
#Contiene las funciones relacionadas al modelo de ML
from model import train_model, evaluate_model

def main():
    config = load_config() #Lee la configuración del proyecto
    df = load_data(config["data_path"]) #Datos crudos
    X, y = preprocess_data(df, config["target_column"])
    #División de datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(X, y, config["test_size"], config["random_state"])

    with mlflow.start_run():
    #Registro de hiperparametro
        mlflow.log_params({
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"]
        })
    #Entrenamiento y evaluación del modelo
        model = train_model(X_train, y_train, config["n_estimators"], config["max_depth"])
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        #Registro del modelo
        mlflow.sklearn.log_model(model, "model")

        # Predicción con los primeros 5 ejemplos del conjunto de prueba
        y_pred = model.predict(X_test[:10])
        print("Predicciones para las primeras 10 muestras:")
        for i, pred in enumerate(y_pred):
            print(f"Muestra {i+1}: Predicción = {pred:.2f}, Real = {y_test.iloc[i]:.2f}")

if __name__ == "__main__":
    main()
