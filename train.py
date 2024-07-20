import torch
from tqdm import tqdm  # progress bar
from typing import Tuple
from IPython import display
from torch import nn, optim
from datetime import datetime
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from emonet.models.fer_emonet import FerEmonet
from torchvision.transforms import functional as F
from scheduler import CosineAnnealingWithWarmRestartsLR as LearningRateScheduler


class Trainer:
    def __init__(self, model, training_dataloader, validation_dataloader, testing_dataloader, execution_name, lr,
                 output_dir, max_epochs, early_stopping_patience, min_delta):
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
        self.scheduler = LearningRateScheduler(self.optimizer, initial_lr=self.lr)  # Initialize the LR scheduler
        self.criterion = nn.CrossEntropyLoss()  # Loss function
        self.counter = 0  # Counter for early stopping
        self.best_loss = float('inf')  # Initialize the best loss for early stopping
        self.early_stop = False  # Flag for early stopping
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        # Move the model to the appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def plot_progress(self):
        # Clear the current figure and display the plot
        display.clear_output(wait=True)
        plt.figure(figsize=(15, 5))

        # Plot training and validation loss
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot validation accuracy
        plt.subplot(1, 3, 2)
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()

        # Plot training accuracy
        plt.subplot(1, 3, 3)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()

        plt.tight_layout()
        display.display(plt.gcf())
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
        # Validate the model on the validation dataset
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in self.validation_dataloader:
                inputs, labels = batch
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        validation_loss = total_loss / len(self.validation_dataloader)
        accuracy = correct / len(self.validation_dataloader.dataset)
        print(f'Validation Loss: {validation_loss}, Accuracy: {accuracy}')
        return validation_loss, accuracy

    def train(self):
        # Train the model
        self.model.train()  # Set the model to training mode
        for epoch in range(self.max_epochs):
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
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                # Update the progress bar
                progress_bar.set_postfix({'loss': f'{total_loss / (progress_bar.n + 1):.4f}'})

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
            self.scheduler.step()

            if self.check_early_stopping(validation_loss):
                print(f"Validation Loss did not improve for {self.early_stopping_patience} epochs. "
                      f"Early stopping triggered.")
                break

    def test(self):
        # Test the model on the testing dataset
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0
        correct = 0
        with torch.no_grad():  # No need to compute gradients
            for batch in self.testing_dataloader:
                inputs, labels = batch
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        test_loss = total_loss / len(self.testing_dataloader)
        accuracy = correct / len(self.testing_dataloader.dataset)
        print(f'Test Loss: {test_loss}, Accuracy: {accuracy}')

    def save_model(self):
        # Save the trained model
        torch.save(self.model.state_dict(), f'{self.output_dir}/{self.execution_name}_trained.pth')

    def run(self):
        # Run the training, validation, testing, and save the model
        self.train()
        self.plot_progress()
        if not self.early_stop:
            self.test()
            self.save_model()


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
        GrayscaleToRGB()  # Convert grayscale images to RGB
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Common ImageNet normalization
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


# Set up command-line arguments for the training script
def set_arguments_for_train(arg_parser: ArgumentParser) -> None:
    # Define all arguments for the Emonet training script
    arg_parser.add_argument("--dataset-path", type=str, default="fer2013", help="Path to the dataset")
    arg_parser.add_argument("--output-dir", type=str, default="out", help="Path where the best model will be saved")
    arg_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    arg_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    arg_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    arg_parser.add_argument("--early_stopping_patience", type=int, default=4, help="Early Stopping")
    arg_parser.add_argument("--min_delta", type=float, default=0.001, help="Min delta of validation loss for ES")
    arg_parser.add_argument("--num-workers", type=int, default=1,
                            help="The number of subprocesses to use for data loading."
                                 "0 means that the data will be loaded in the main process.")
    arg_parser.add_argument('--emonet_classes', type=int, default=5, choices=[5, 8],
                            help='Number of emotional classes to test the model on. Please use 5 or 8.')


if __name__ == "__main__":
    parser = ArgumentParser(description="Train our version of Emonet on Fer2013")

    set_arguments_for_train(parser)

    args = parser.parse_args()
    print(args)

    # Generate a unique identifier for this training session or model save file
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exec_name = f"Emonet_{args.emonet_classes}_{current_time}"

    # Load and transform datasets, then create DataLoaders for training, validation, and testing
    train_dataset, val_dataset, test_dataset = load_and_transform_datasets(args.dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the FerEmonet model
    fer_emonet_model = FerEmonet(emonet_classes=args.emonet_classes)

    Trainer(
        model=fer_emonet_model,
        training_dataloader=train_loader,
        validation_dataloader=val_loader,
        testing_dataloader=test_loader,
        execution_name=exec_name,
        lr=args.lr,
        output_dir=args.output_dir,
        max_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta
    ).run()

    # Save the model
    torch.save(fer_emonet_model.state_dict(), f'emonet_{args.emonet_classes}_trained.pth')
