import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

def load_config():
    with open("src/config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df, target_column):
    df = df.dropna()  
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Codificar variables categóricas
    X = pd.get_dummies(X)
    
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
