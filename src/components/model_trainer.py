import os
import sys
from dataclasses import dataclass
import numpy as np

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
     try:
        train_array = np.array(train_array, dtype=np.float64)
        test_array  = np.array(test_array,  dtype=np.float64)

# TEMP DEBUG - paste output here
        print(f"Array shape: {train_array.shape}")
        for i in range(train_array.shape[1]):
          col = train_array[:, i]
          print(f"Col {i}: min={col.min():.3f}, max={col.max():.3f}, mean={col.mean():.3f}, nulls={np.isnan(col).sum()}")

        # --- Find log_price column (values between 10 and 16) ---
        target_col = None
        for i in range(train_array.shape[1]):
            col = train_array[:, i]
            if col.mean() > 11 and col.max() < 17 and col.max() > 10:
                target_col = i
                print(f"Found target at column {i}, mean={col.mean():.3f}")
                break

        if target_col is None:
            raise CustomException("Could not find log_price column", sys)

        X_train = np.delete(train_array, target_col, axis=1)
        y_train = train_array[:, target_col]
        X_test  = np.delete(test_array,  target_col, axis=1)
        y_test  = test_array[:, target_col]

        print(f"X_train shape: {X_train.shape}, y_train sample: {y_train[:3]}")

        models = {
            "Random Forest":    RandomForestRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor":     XGBRegressor(),
        }
        params = {
            "Random Forest":    {'n_estimators': [8,16,32,64,128,256]},
            "Linear Regression": {},
            "XGBRegressor":     {
                'learning_rate': [.1, .01, .05],
                'n_estimators':  [8, 16, 32, 64, 128, 256]
            },
        }

        model_report: dict = evaluate_models(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            models=models,   param=params
        )

        print("\n" + "="*20)
        print("MODEL PERFORMANCE REPORT:")
        for name, score in model_report.items():
            print(f"  {name}: {score:.4f}")
        print("="*20 + "\n")

        best_model_score = max(model_report.values())
        best_model_name  = max(model_report, key=model_report.get)
        best_model       = models[best_model_name]

        if best_model_score < 0.6:
            raise CustomException(f"Best score too low: {best_model_score:.4f}", sys)

        logging.info(f"Best model: {best_model_name} with R2={best_model_score:.4f}")

        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )

        predicted        = best_model.predict(X_test)
        real_predictions = np.expm1(predicted)
        real_actuals     = np.expm1(y_test)

        r2_square = r2_score(real_actuals, real_predictions)
        return r2_square

     except Exception as e:
        raise CustomException(e, sys)

            
        