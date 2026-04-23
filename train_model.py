import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def main():
    # Загружаем данные
    df = pd.read_csv('df_clear.csv')
    print(f'Data loaded: {df.shape}')

    # Подготавливаем признаки
    X = df.drop(columns=['Price(euro)'])
    y = df['Price(euro)']
    X = X.select_dtypes(include=[np.number])

    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучаем модель
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Оцениваем качество
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')

    # Логируем в mlflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment('linear_model_cars')

    with mlflow.start_run() as run:
        mlflow.log_param('model_type', 'LinearRegression')
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('r2', r2)
        mlflow.sklearn.log_model(model, 'model')

        # ✅ ПРАВИЛЬНЫЙ URI
        model_uri = f'runs:/{run.info.run_id}/model'
        print(f'Model URI: {model_uri}')

        with open('best_model.txt', 'w') as f:
            f.write(model_uri)

    print("Model training completed!")

if __name__ == "__main__":
    main()
