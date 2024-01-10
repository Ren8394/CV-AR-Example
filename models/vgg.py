import torch
import torch.nn as nn

from torchvision.models import vgg16, VGG16_Weights

class VGG16(nn.Module):
    """
    VGG16 model
    input: 3 x 160 x 160
    output: 15
    VGG16 + 2-layer FC
    """
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.vgg16 = vgg16(weights=VGG16_Weights.DEFAULT)
        for param in self.vgg16.features.parameters():
            param.requires_grad = False
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.vgg16.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.vgg16(x)

        return x
    
if __name__ == "__main__":
    from torchinfo import summary
    
    model = VGG16(num_classes=15)
    # summary(model, (16, 3, 160, 160))

    print(model(torch.rand(1, 3, 160, 160)))