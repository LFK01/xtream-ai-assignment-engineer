# Standard Library
from logging import Logger
from abc import ABC, abstractmethod
import datetime
import os
import pickle
import json

# Data Science Tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Project imports
from src.data_utils.preprocess import DataProcessor
from src.model.neural_network import NeuralNetworkRegressionModel
from src.utils.consts import SAVED_MODELS_DIR, LOG_DIR, MODEL_FILENAME, PCA_PICKLE_FILENAME, MEAN_STD_DEV_FILENAME

SEED = 1234
TRAIN_SIZE = 0.7
VAL_SIZE = 0.2
TEST_SIZE = 0.1

LEARNING_RATE = 10 ** -3
BATCH_SIZE = 16
PATIENCE = 10
NUM_EPOCHS = 10 ** 3

torch.manual_seed(SEED)


class PricePredictor(ABC):
    def __init__(self,
                 headers_dict: dict,
                 logger: Logger,
                 data: pd.DataFrame,
                 reduce_dimensions: bool = True,
                 ) -> None:
        self.headers_dict = headers_dict
        self.data = data
        self.preprocessor = DataProcessor(reduce_dimensions=reduce_dimensions)

        self.logger = logger

    @abstractmethod
    def prepare_datasets(self) -> None:
        self.data = self.preprocessor.process_data(df=self.data)

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def test(self) -> None:
        pass

    @abstractmethod
    def infer(self,
              input_df: pd.DataFrame,
              ) -> list[int]:
        pass


class NeuralNetworkPredictor(PricePredictor):
    def __init__(self,
                 headers_dict: dict,
                 hidden_size: int,
                 logger: Logger,
                 data: pd.DataFrame = None,
                 learning_rate: float = LEARNING_RATE,
                 batch_size: int = BATCH_SIZE,
                 patience: int = PATIENCE,
                 num_epochs: int = NUM_EPOCHS,
                 reduce_dimensions: bool = True,
                 ) -> None:
        super().__init__(headers_dict=headers_dict,
                         data=data,
                         logger=logger,
                         reduce_dimensions=reduce_dimensions,
                         )
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.patience = patience
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.model = None
        self.train_split, self.val_split, self.test_split = None, None, None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        self.model_directory = None
        self.criterion = nn.L1Loss()

    def prepare_datasets(self) -> None:
        super().prepare_datasets()

        # https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
        self.train_split, self.val_split, self.test_split = np.split(self.data.sample(frac=1, random_state=SEED),
                                                                     [int(TRAIN_SIZE * len(self.data)),
                                                                      int((TRAIN_SIZE + VAL_SIZE) * len(self.data))])

        self.logger.debug(f'train size: {self.train_split.shape[0]}, val size: {self.val_split.shape[0]}, '
                          f'test size: {self.test_split.shape[0]}')
        self.logger.debug(f'Train Split Normalized DataFrame Head:\n{self.train_split.head()}')

        # Dataset definitions (input and target variables)
        self.X_train = torch.tensor(self.train_split[self.preprocessor.feature_cols].values,  # input data
                                    dtype=torch.float32)
        self.y_train = torch.tensor(self.train_split[self.preprocessor.target_col].values,  # target variable
                                    dtype=torch.float32)

        self.X_val = torch.tensor(self.val_split[self.preprocessor.feature_cols].values, dtype=torch.float32)
        self.y_val = torch.tensor(self.val_split[self.preprocessor.target_col].values, dtype=torch.float32)

        self.X_test = torch.tensor(self.test_split[self.preprocessor.feature_cols].values, dtype=torch.float32)
        self.y_test = torch.tensor(self.test_split[self.preprocessor.target_col].values, dtype=torch.float32)

        # Create DataLoader for batch processing
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        test_dataset = TensorDataset(self.X_test, self.y_test)

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self) -> None:
        self.model = NeuralNetworkRegressionModel(self.X_train.shape[1], self.hidden_size, output_size=1)
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate)

        # Initialize variables for early stopping
        best_val_loss = float('inf')
        best_model_state = None  # Initialize variable to store best model state

        counter = 0  # Counter for epochs without improvement

        # Training loop with validation
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for inputs, targets in self.train_dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                # Add square root to read more meaningful results
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in self.val_dataloader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            # Print training and validation loss
            train_loss /= len(self.train_dataloader.dataset)
            val_loss /= len(self.val_dataloader.dataset)
            self.logger.info(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, '
                             f'Val Loss: {val_loss:.4f}')

            # Check for improvement in validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()  # Save current model state
                counter = 0
            else:
                counter += 1

            # Check if early stopping criteria met
            if counter >= self.patience:
                self.logger.info('Early stopping: No improvement in validation loss.')
                break

        self.save_model(best_val_loss=best_val_loss,
                        best_model_state=best_model_state)

    def test(self) -> None:
        super().test()

        # Load the best model state
        self.model = NeuralNetworkRegressionModel(self.X_train.shape[1], self.hidden_size, output_size=1)
        self.model.load_state_dict(torch.load(os.path.join(self.model_directory, MODEL_FILENAME)))

        # Evaluate model on test set
        self.model.eval()
        test_loss = 0.0

        # Save predictions on test set to plot the comparison graph
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_dataloader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

                all_predictions.extend(outputs.squeeze().tolist())
                all_targets.extend(targets.tolist())

        # Calculate test loss
        test_loss /= len(self.test_dataloader.dataset)
        self.logger.debug(f'Test Loss: {test_loss:.4f}')

        # Convert predictions and targets to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        # Plot predictions vs. targets
        plt.figure(figsize=(8, 8))
        plt.scatter(all_targets, all_predictions, alpha=0.5)
        plt.plot([min(self.test_split[self.preprocessor.target_col].values),
                  max(self.test_split[self.preprocessor.target_col].values)],
                 [min(self.test_split[self.preprocessor.target_col].values),
                  max(self.test_split[self.preprocessor.target_col].values)],
                 color='red')  # Plotting the diagonal line
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs. Predicted Values')
        plt.grid(True)
        now_date = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        plt.savefig(fname=os.path.join(LOG_DIR, f'{now_date}_graph_test_results.png'))

    def infer(self,
              input_df: pd.DataFrame,
              ) -> list[int]:
        self.check_model_directory()
        self.load_mean_std_values()
        self.load_pca()

        df = input_df.copy()

        df = self.preprocessor.convert_categorical(df=df)
        df = self.preprocessor.reduce_dimensions(df=df, pca=self.preprocessor.pca)
        df = self.preprocessor.normalize_values(df=df)

        self.X_test = torch.tensor(df[self.preprocessor.feature_cols].values, dtype=torch.float32)
        test_dataset = TensorDataset(self.X_test)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.load_model()

        self.model.eval()
        # Save predictions to return the inferred values
        all_predictions = []

        with torch.no_grad():
            for inputs, in self.test_dataloader:
                outputs = self.model(inputs)
                if len(outputs) > 1:
                    all_predictions.extend(outputs.squeeze().tolist())
                else:
                    all_predictions.append(outputs.squeeze().tolist())

        return [int(pred) for pred in all_predictions]

    def save_model(self,
                   best_val_loss: float,
                   best_model_state):
        # Create directory where model, pca compressor and dataset mean and std dev values will be saved
        directory_name = (f'{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}'
                          f'_model_loss_{int(best_val_loss)}')
        # Create full path
        self.model_directory = os.path.join(SAVED_MODELS_DIR, directory_name)
        # Create directory
        os.mkdir(self.model_directory)
        # Create full path for the model file
        model_filepath = os.path.join(self.model_directory, MODEL_FILENAME)

        # Save the best model state
        torch.save(best_model_state, model_filepath)

        # Save the pca associated to the data used for this model
        with open(os.path.join(self.model_directory, PCA_PICKLE_FILENAME), 'wb') as pickle_file:
            pickle.dump(obj=self.preprocessor.pca, file=pickle_file)

        # Save the mean and std dev values
        with open(os.path.join(self.model_directory, MEAN_STD_DEV_FILENAME), 'w') as json_file:
            json.dump(obj=self.preprocessor.mean_std_dict,
                      fp=json_file,
                      sort_keys=True,
                      indent=4)

    def check_model_directory(self) -> str:
        if self.model_directory and os.path.exists(self.model_directory):
            model_filepath = os.path.join(self.model_directory, MODEL_FILENAME)
        else:
            if len(os.listdir(SAVED_MODELS_DIR)) > 0:
                # If there is more than one saved model pick the one with the lowest saved validation loss

                # Split the directories filenames by underscores and take only the loss value,
                # order the directories by their loss and take the minimum
                self.model_directory = os.path.join(SAVED_MODELS_DIR,
                                                    min(os.listdir(SAVED_MODELS_DIR),
                                                        key=lambda x: int(x.split('_')[-1])
                                                        )
                                                    )

                model_filepath = os.path.join(self.model_directory, MODEL_FILENAME)
            else:
                self.logger.error('If there are no model saved you need to run a training before executing infer '
                                  'or upload a compatible model to the saved models folder.')
                raise PermissionError('If there are no model saved you need to run a training before executing infer '
                                      'or upload a compatible model to the saved models folder.')

        return str(model_filepath)

    def load_model(self):
        # Load the best model state
        self.model = NeuralNetworkRegressionModel(self.X_test.shape[1], self.hidden_size, output_size=1)
        self.model.load_state_dict(torch.load(os.path.join(self.model_directory, MODEL_FILENAME)))

    def load_pca(self):
        # Load the pca compressor
        with open(os.path.join(self.model_directory, PCA_PICKLE_FILENAME), 'rb') as pickle_load_file:
            self.preprocessor.pca = pickle.load(pickle_load_file)

    def load_mean_std_values(self):
        # Load std and mean values to normalize data
        with open(os.path.join(self.model_directory, MEAN_STD_DEV_FILENAME), 'r') as file:
            self.preprocessor.mean_std_dict = json.load(file)
