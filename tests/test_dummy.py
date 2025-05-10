import pandas as pd

def test_data_loading():
    df = pd.read_csv("data/Life Expectancy Data.csv")
    assert not df.empty, "El archivo CSV está vacío"