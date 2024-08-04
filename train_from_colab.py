# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install necessary libraries
!pip install torch pandas matplotlib seaborn


import os
import torch
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm  # progress bar
from typing import Tuple
from torch import nn, optim
from datetime import datetime
import matplotlib.pyplot as plt
from IPython.display import display, Image
from torchvision import datasets, transforms
from emonet.models.fer_emonet import FerEmonet
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from torchvision.transforms import functional as F
from emonet.models.fer_multihead import FerMultihead
from torch.utils.data import DataLoader, ConcatDataset
from emonet.models.emonet_self_attention import EmonetWithSelfAttention
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from scheduler import CosineAnnealingWithWarmRestartsLR as LearningRateScheduler


class Trainer:
    def __init__(self, model, training_dataloader, validation_dataloader, testing_dataloader, execution_name, lr,
                 output_dir, max_epochs, early_stopping_patience, min_delta):
        # for saving model
        self.best_model_state = None
        self.model = model
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.testing_dataloader = testing_dataloader
        self.execution_name = execution_name
        self.lr = lr
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # Initialize the optimizer (Adam and not SGD)
        self.scheduler = LearningRateScheduler(self.optimizer, t_0=10, t_mult=1, eta_min=0, last_epoch=-1,
                                               initial_lr=self.lr)  # Initialize the LR scheduler
        self.criterion = nn.CrossEntropyLoss()  # Loss function
        self.counter = 0  # Counter for early stopping
        self.best_loss = float('inf')  # Initialize the best loss for early stopping
        self.early_stop = False  # Flag for early stopping
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.precision_list = []
        self.recall_list = []
        self.f1_list = []
        self.confusion_matrices = []

        # Move the model to the appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def plot_accuracy_and_loss(self):
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        plot_path = os.path.join(self.output_dir, f'{self.execution_name}_accuracy_and_loss.png')
        plt.figure(figsize=(20, 10))

        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)  # Save the plot to a file
        display(Image(filename=plot_path))  # Display the saved plot image in the notebook
        self.save_to_google_drive(plt.gcf(), f'{self.execution_name}_accuracy_and_loss.png')
        plt.close()  # Close the figure to prevent it from being displayed inline in the notebook

    def plot_confusion_matrix(self):
        # Calculate confusion matrix for the validation set
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in self.validation_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        conf_matrix = confusion_matrix(all_labels, all_preds)

        plot_path = os.path.join(self.output_dir, f'{self.execution_name}_confusion_matrix.png')
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(plot_path)  # Save the plot to a file
        display(Image(filename=plot_path))  # Display the saved plot image in the notebook
        self.save_to_google_drive(plt.gcf(), f'{self.execution_name}_confusion_matrix.png')
        plt.close()  # Close the figure to prevent it from being displayed inline in the notebook

    def plot_precision_recall_curve(self):
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in self.validation_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
        all_labels = label_binarize(all_labels, classes=[0, 1, 2, 3, 4, 5, 6])  # Adjust based on number of classes

        plot_path = os.path.join(self.output_dir, f'{self.execution_name}_precision_recall_curve.png')
        plt.figure(figsize=(10, 8))
        for i in range(all_labels.shape[1]):
            precision, recall, _ = precision_recall_curve(all_labels[:, i], np.array(all_probs)[:, i])
            plt.plot(recall, precision, label=f'Class {i}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)  # Save the plot to a file
        display(Image(filename=plot_path))  # Display the saved plot image in the notebook
        self.save_to_google_drive(plt.gcf(), f'{self.execution_name}_precision_recall_curve.png')
        plt.close()  # Close the figure to prevent it from being displayed inline in the notebook

    def check_early_stopping(self, validation_loss):
        # Check if early stopping criteria are met
        if self.best_loss - validation_loss > self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.early_stopping_patience:
                self.early_stop = True
                return True
            return False

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in self.validation_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        validation_loss = total_loss / len(self.validation_dataloader)
        accuracy = correct / len(self.validation_dataloader.dataset)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_preds)

        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.f1_list.append(f1)
        self.confusion_matrices.append(conf_matrix)

        print(f'Validation Loss: {validation_loss}, Accuracy: {accuracy}, Precision: {precision}, '
              f'Recall: {recall}, F1 Score: {f1}')
        print(f'Confusion Matrix:\n{conf_matrix}')
        return validation_loss, accuracy

    def train(self):
        # Train the model
        self.model.train()  # Set the model to training mode
        for epoch in range(self.max_epochs):
            print(f'Epoch {epoch + 1}/{self.max_epochs}, Learning Rate: {self.optimizer.param_groups[0]["lr"]}')
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0
            progress_bar = tqdm(self.training_dataloader, desc=f'Epoch {epoch + 1}/{self.max_epochs}', unit="batch")
            for batch in self.training_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # clears old gradients from the last step,
                # necessary because gradients accumulate by default for every backpropagation pass
                self.optimizer.zero_grad()
                # Forward pass, the model outputs the predicted labels - raw scores (logits) for each class
                outputs = self.model(inputs)
                # calculates the loss between the predicted outputs and the actual labels using cross-entropy loss
                # combines nn.LogSoftmax and nn.NLLLoss in one single class, therefore no need to apply softmax
                loss = self.criterion(outputs, labels)
                # Backward pass, computes the gradient of the loss with respect to the model parameters
                loss.backward()
                # updates the model parameters based on the gradients calculated during backpropagation
                self.optimizer.step()
                current_loss = loss.item()
                total_loss += current_loss

                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                # Update progress bar with loss value
                progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
                progress_bar.update(1)  # Manually increment the progress bar by one step

            progress_bar.close()

            train_accuracy = correct_predictions / total_predictions

            validation_loss, validation_accuracy = self.validate()
            train_loss = total_loss / len(self.training_dataloader)

            print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {validation_loss}, '
                  f'Train Accuracy: {train_accuracy}, Validation Accuracy: {validation_accuracy}')

            # Append metrics to their respective lists
            self.train_losses.append(train_loss)
            self.val_losses.append(validation_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(validation_accuracy)

            # Update Learning Rate
            # self.scheduler.step(epoch)
            if validation_loss < self.best_loss:
                self.best_model_state = self.model.state_dict()

            if self.check_early_stopping(validation_loss):
                print(f"Validation Loss did not improve for {self.early_stopping_patience} epochs. "
                      f"Early stopping triggered.")
                break

        # at the emd of training we load the best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def test(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in self.testing_dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        test_loss = total_loss / len(self.testing_dataloader)
        accuracy = correct / len(self.testing_dataloader.dataset)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_labels, all_preds)

        print(f'Test Loss: {test_loss}, Accuracy: {accuracy}, Precision: {precision}, '
              f'Recall: {recall}, F1 Score: {f1}')
        print(f'Confusion Matrix:\n{conf_matrix}')

    def save_model_and_results(self):
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Save the trained model locally
        local_model_path = f'{self.output_dir}/{self.execution_name}_trained.pth'
        torch.save(self.model.state_dict(), local_model_path)
        print(f'Model saved to {local_model_path}')

        # Create a DataFrame with the training and validation metrics
        results_df = pd.DataFrame({
            'Epoch': list(range(1, len(self.train_losses) + 1)),
            'Train Loss': self.train_losses,
            'Validation Loss': self.val_losses,
            'Train Accuracy': self.train_accuracies,
            'Validation Accuracy': self.val_accuracies
        })

        # Print the DataFrame
        print(results_df)

        # Define the local path to save the CSV file
        csv_file_path = os.path.join(self.output_dir, f'{self.execution_name}_training_results.csv')

        # Save the DataFrame as a CSV file locally
        results_df.to_csv(csv_file_path, index=False)
        print(f'Results saved to {csv_file_path}')
        # Save to Google Drive if a folder ID is provided
        self.save_to_google_drive(local_model_path, f'{self.execution_name}_trained.pth')
        self.save_to_google_drive(csv_file_path, f'{self.execution_name}_training_results.csv')

    def save_to_google_drive(self, data, filename):

        drive_path = f'/content/drive/My Drive/Facial-Expression-Recognition-Emonet/{self.execution_name}'
        os.makedirs(drive_path, exist_ok=True)

        full_path = os.path.join(drive_path, filename)

        if isinstance(data, str):  # It's a file path (model or DataFrame)
            shutil.copyfile(data, full_path)  # Use shutil.copyfile() to copy
        elif isinstance(data, plt.Figure):  # It's a matplotlib Figure
            data.savefig(full_path)
        else:
            print(f"Unsupported data type for saving: {type(data)}")

        print(f'Saved to Google Drive: {full_path}')

    def run(self):
        # Run the training, validation, testing, and save the model
        self.train()
        self.plot_accuracy_and_loss()
        self.plot_confusion_matrix()
        self.plot_precision_recall_curve()
        self.test()
        self.save_model_and_results()


class GrayscaleToRGB:
    # Convert a grayscale image to RGB by repeating the grayscale channel thrice.
    def __call__(self, img):
        """
        Args: img (PIL Image or Tensor): Image to be converted to RGB.
        Returns: PIL Image or Tensor: RGB image.
        """
        return F.to_tensor(F.to_pil_image(img).convert("RGB"))

    def __repr__(self):
        return self.__class__.__name__ + '()'


# Define transformations for the training, validation, and testing datasets
def dataset_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),  # Resize to 256x256
        transforms.ToTensor(),  # Convert to tensor
        GrayscaleToRGB(),  # Convert grayscale images to RGB
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Common ImageNet normalization
    ])


def dataset_transform_mma() -> transforms.Compose:
    # TODO: change RGB
    return transforms.Compose([
        transforms.Resize(256),  # Resize to 256x256
        transforms.ToTensor(),  # Convert to tensor
        GrayscaleToRGB(),  # Convert grayscale images to RGB
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Common ImageNet normalization
    ])


# Load datasets from the specified path and apply transformations
def load_dataset(dataset_path: str, subset: str, transform: transforms.Compose) -> datasets.VisionDataset:
    return datasets.ImageFolder(root=f'{dataset_path}/{subset}', transform=transform)


# Main function to load and transform datasets for training, validation, and testing
def load_and_transform_datasets(dataset_path: str) -> Tuple[
    datasets.VisionDataset, datasets.VisionDataset, datasets.VisionDataset]:
    train_dataset = load_dataset(dataset_path, 'train', dataset_transform())
    val_dataset = load_dataset(dataset_path, 'val', dataset_transform())
    test_dataset = load_dataset(dataset_path, 'test', dataset_transform())

    print(f'Using {len(train_dataset)} images for training.')
    print(f'Using {len(val_dataset)} images for evaluation.')
    print(f'Using {len(test_dataset)} images for testing.')

    return train_dataset, val_dataset, test_dataset


def load_and_transform_datasets_mma(dataset_path: str) -> Tuple[
    datasets.VisionDataset, datasets.VisionDataset, datasets.VisionDataset]:
    train_dataset = load_dataset(dataset_path, 'train', dataset_transform_mma())
    val_dataset = load_dataset(dataset_path, 'valid', dataset_transform_mma())
    test_dataset = load_dataset(dataset_path, 'test', dataset_transform_mma())

    print(f'Using {len(train_dataset)} images for training.')
    print(f'Using {len(val_dataset)} images for evaluation.')
    print(f'Using {len(test_dataset)} images for testing.')

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    args = {'emonet_classes': 5, 'final_layer_type': 1, 'attention': 'Default', 'epochs': 30,
            'num_workers': 1, 'lr': 1e-3, 'output_dir': 'trained_models_folder',
            'early_stopping_patience': 5, 'min_delta': 0.001, 'fer_dataset_path': '/content/fer2013',
            'mma_dataset_path':'/content/Facial-Expression-Recognition-Emonet/mma/MMAFEDB', 'batch_size': 32}
    # Generate a unique identifier for this training session or model save file
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exec_name = f"Emonet_{args['emonet_classes']}_{current_time}"

    # Load and transform datasets, then create DataLoaders for training, validation, and testing
    train_dataset, val_dataset, test_dataset = load_and_transform_datasets(args['fer_dataset_path'])
    train_dataset_mma, val_dataset_mma, test_dataset_mma = load_and_transform_datasets_mma(args['mma_dataset_path'])

    combined_dataset_train = ConcatDataset([train_dataset, train_dataset_mma])
    combined_dataset_val = ConcatDataset([val_dataset, val_dataset_mma])
    combined_dataset_test = ConcatDataset([test_dataset, test_dataset_mma])

    train_loader = DataLoader(combined_dataset_train, batch_size=args['batch_size'], shuffle=True,
                              num_workers=args['num_workers'])
    val_loader = DataLoader(combined_dataset_val, batch_size=32, shuffle=False)
    test_loader = DataLoader(combined_dataset_test, batch_size=32, shuffle=False)

    # Initialize the Emonet model
    if args['attention'] == 'Self-Attention':
        fer_emonet_model = EmonetWithSelfAttention(emonet_classes=args['emonet_classes'])
    elif args['attention'] == 'Multi-Head-Attention':
        fer_emonet_model = FerMultihead(emonet_classes=args['emonet_classes'])
    else:
        fer_emonet_model = FerEmonet(emonet_classes=args['emonet_classes'], final_layer_type=args['final_layer_type'])

    Trainer(
        model=fer_emonet_model,
        training_dataloader=train_loader,
        validation_dataloader=val_loader,
        testing_dataloader=test_loader,
        execution_name=exec_name,
        lr=args['lr'],
        output_dir=args['output_dir'],
        max_epochs=args['epochs'],
        early_stopping_patience=args['early_stopping_patience'],
        min_delta=args['min_delta'],
    ).run()
