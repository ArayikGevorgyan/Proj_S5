from src.data_preparation import load_data, clean_data, summarize_data

def test_data_loading():
    df = load_data("data/sample_patients.csv")
    assert not df.empty, "Dataset should not be empty"

def test_clean_data():
    df = load_data("data/sample_patients.csv")
    df_clean = clean_data(df)
    assert "BMI" in df_clean.columns
    assert df_clean["BMI"].isnull().sum() == 0
