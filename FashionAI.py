import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

trainingData = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)

testData = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)

batchSize = 64 #each element in the loader will return 64 features/labels

train_dataLoader = DataLoader(trainingData, batch_size=batchSize)
test_dataLoader = DataLoader(testData, batch_size=batchSize)

for x,y in test_dataLoader:
    print(f"Shape of X [N,C,H,W]: {x.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#data has been loaded

class neuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() #turn 2d image to 1d vector
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = neuralNetwork()
print(model)

loss_fn = nn.CrossEntropyLoss() #loss function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x,y) in enumerate(dataloader):
        #compute error
        pred = model(x)
        loss = loss_fn(pred, y)

        #backpropogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current = (batch+1)*len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    numBatches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= numBatches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#epoch is an iteration
epochs = 5
for t in range(epochs):
    print(f"epoch {t+1}\n-------------------")
    train(train_dataLoader, model, loss_fn, optimizer)
    test(test_dataLoader, model, loss_fn)
print("done")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
c = 0 #num correct
n = 15 #num of example tests
for i in range(n):
    x, y = testData[i][0], testData[i][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        if(predicted == actual):
            c += 1
print(f"accuracy: {c}/{n}")
