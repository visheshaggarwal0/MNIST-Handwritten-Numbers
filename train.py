import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from model import MNISTCNN

def train_model(epochs=15, batch_size=128, learning_rate=0.001, patience=3):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Define image preprocessing/transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST standard normalization constants
    ])
    
    # 2. Download and load datasets
    print("Downloading and preparing datasets...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Instantiate model, loss, and optimizer
    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. Training Loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_loss = float('inf')
    patience_counter = 0
    model_path = 'mnist_cnn.pth'
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if (batch_idx + 1) % 200 == 0:
                print(f"Epoch [{epoch}/{epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100. * correct / total
        
        # Evaluation Phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
                
        epoch_test_loss = test_loss / len(test_loader.dataset)
        epoch_test_acc = 100. * test_correct / test_total
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_loss'].append(epoch_test_loss)
        history['test_acc'].append(epoch_test_acc)
        
        print(f"==> Epoch {epoch} summary | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.2f}%")
        
        # Early stopping logic & best model saving
        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss decreased. Saved best model checkpoint to {model_path}")
        else:
            pvariance_val = epoch_test_loss - best_loss
            patience_counter += 1
            print(f"Validation loss did not decrease (diff: +{pvariance_val:.4f}). Early stopping counter: {patience_counter} of {patience}")
            
        if patience_counter >= patience:
            print("Early stopping triggered! Training halted.")
            break
            
    # 5. Plot & save training history
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='#636EFA', marker='o')
    plt.plot(history['test_loss'], label='Test Loss', color='#EF553B', marker='o')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='#00CC96', marker='o')
    plt.plot(history['test_acc'], label='Test Acc', color='#AB63FA', marker='o')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    print("Training history plot saved as training_history.png")
    
    return history

if __name__ == "__main__":
    train_model(epochs=3)

