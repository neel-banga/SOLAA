import torch
from torchvision import transforms,datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

data_transforms = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# Load the dataset
trainset = datasets.ImageFolder('CONV_DATA', transform=data_transforms)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=7, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(123904, 7)
    
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1) # Flatten the tensor along second dimension
        x = self.fc1(x)
        return x



net = Net()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
EPOCHS = 100

def train_model():
    for epoch in range(EPOCHS):
        for batch in train_loader:
            X,y = batch
            yhat = net(X)
            loss = loss_fn(yhat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch:{epoch} loss is {loss.item()}")
    torch.save(net.state_dict(), 'model.pth')

#train_model()



# net.load_state_dict(torch.load('model.pth'))
# validation()


def check(image):
    
    img = Image.open(image)
    img_transformed = data_transforms(img)
    img_batch = img_transformed.unsqueeze(0)  # Add a batch dimension

    output = net(img_batch)
    print(output)
    predicted_class = torch.argmax(output)
    print(predicted_class)
    print("Predicted class:", predicted_class.item())
    if predicted_class.item() == 1:
        return 'NEEL'
    else:
        return 'VEDANT'


net.load_state_dict((torch.load('model.pth')))
net.eval()
