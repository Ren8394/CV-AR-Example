import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        """
        CNN model
        input: 3 x 160 x 160
        output: 15
        2-layer CNN + 2-layer FC
        """
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 40 * 40, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        
        return x
    
if __name__ == "__main__":
    from torchinfo import summary
    
    model = CNN(num_classes=15)
    summary(model, (16, 3, 160, 160))