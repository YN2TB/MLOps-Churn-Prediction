import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.config import models, param_grids, ModelTrainerConfig
from src.components.preprocessor import Preprocessor


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def _prepare_target(self, y):
        # ...existing code...
        # Convert DataFrame with one column -> Series
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Target y must have exactly one column for classification.")
            y = y.iloc[:, 0]

        # Convert to 1D array
        y = np.ravel(y)

        # Normalize string labels if needed
        if y.dtype.kind in {"O", "U", "S"}:
            y_series = pd.Series(y).astype(str).str.strip().str.lower()
            binary_map = {
                "yes": 1, "no": 0,
                "true": 1, "false": 0,
                "1": 1, "0": 0
            }
            if y_series.isin(binary_map.keys()).all():
                y = y_series.map(binary_map).to_numpy(dtype=int)
            else:
                y = pd.factorize(y_series)[0]

        return y

    def train_model(self, model, model_name, param_grid, X_train, y_train, X_test, y_test):
        logging.info(f"Training {model_name}")

        y_train = self._prepare_target(y_train)
        y_test = self._prepare_target(y_test)

        if os.path.exists(os.path.join(self.model_trainer_config.trained_model_file_path, f"{model_name}_params.pkl")):
            with open(os.path.join(self.model_trainer_config.trained_model_file_path, f"{model_name}_params.pkl"), "rb") as f:
                best_params = pickle.load(f)
                for key, value in best_params.items():
                    best_params[key] = [value]
        else:
            logging.warning(f"No best parameters found for {model_name}, using default parameter grid.")
            best_params = param_grid

        grid_search = GridSearchCV(
            model,
            best_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            error_score='raise'
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logging.info(f"Best {model_name} model: {best_model}")
        logging.info(f"Best {model_name} parameters: {best_params}")
        logging.info(f"Best {model_name} score: {best_score}")

        save_object(
            file_path=os.path.join(self.model_trainer_config.trained_model_file_path, f"{model_name}.pkl"),
            obj=best_model
        )
        save_object(
            file_path=os.path.join(self.model_trainer_config.trained_model_file_path, f"{model_name}_params.pkl"),
            obj=best_params
        )

        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"{model_name} test accuracy: {test_accuracy}")

        return best_model, best_params, best_score, test_accuracy

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        # Remove target leakage if Churn is still in features
        if isinstance(X_train, pd.DataFrame):
            for col in ["Churn", "churn"]:
                if col in X_train.columns:
                    X_train = X_train.drop(columns=[col])
                if isinstance(X_test, pd.DataFrame) and col in X_test.columns:
                    X_test = X_test.drop(columns=[col])

        preprocessor = Preprocessor()
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        model_results = {}
        for model_name, model in models.items():
            param_grid = param_grids.get(model_name, {})
            best_model, best_params, best_score, test_accuracy = self.train_model(
                model, model_name, param_grid, X_train, y_train, X_test, y_test
            )
            model_results[model_name] = {
                'best_model': best_model,
                'best_params': best_params,
                'best_score': best_score,
                'test_accuracy': test_accuracy
            }

        save_object(
            file_path=os.path.join(self.model_trainer_config.trained_model_file_path, "preprocessor.pkl"),
            obj=preprocessor
        )

        return model_results
    
    def handle_training_exception(self, exception, model_name="Unknown"):
        """Handle and log training exceptions with custom error details."""
        error_message = f"Error occurred during training of {model_name}: {str(exception)}"
        raise CustomException(error_message, sys)