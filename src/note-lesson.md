# Lesson Noted of PyTorch Specialization Course

The ML Pipeline

- Data Ingestion
  - raw data collection
- Data Preprocessing
  - data access
  - data quality
  - data efficiency
- Modeling
- Training
- Evaluation
- Deployment

## Data Ingestion

after raw data collection, we need to load data with batching (small chunk of data)

## Data Preprocessing

1. transforms

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])
```

`ToTensor` mean convert python array to tensor and fall value around 0 and 1.

`Normalize` mean normalize the data to standard distribution.

2.  Dataset

```python
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

some_data = SomeDataset(root='./data', train=True, download=True, transform=transform)

first_Dataset = dataset[0]
```

dataset have several functions :

    - set where data on disk
    - how to load specific sample
    - how many total samples
    - how to apply transform

`root` mean where data on disk

`train` mean data get split into train and test or not.

`download` mean download data if not exist

`transform` mean transform data

`first_Dataset` mean how to retrive the data from dataset

3. DataLoader

```python
DataLoader(dataset, batch_size=64, shuffle=True)
```

`batch_size` mean how many data in one batch

`shuffle` mean shuffle the data or not

`DataLoader` mean function to load data with batching

## Model

```python
class ExampleModel(nn.Module):
    def __init__(self):
      super().__init__()
      self.layer1 = nn.Linear(in_features=1, out_features=20)
      self.relu = nn.ReLU()
      self.layer2 = nn.Linear(in_features=20, out_features=1)

    def forward(self, x):
      x = self.layer1(x)
      x = self.relu(x)
      x = self.layer2(x)
      return x

model = ExampleModel()
output = model(data)

```

`super().__init__()` will track all of learnable parameters in model
`forward` is function that define flow of model layer and data flow through model, also call automatically when call model

## Training Model

```python
# this is standart training loop
for epoch in range(epochs):
  optimizer.zero_grad()
  output = model(data)
  loss = loss_fn(output, labels)
  loss.backward()
  optimizer.step()
```

`optimizer.zero_grad()` mean reset gradient to zero

`loss.backward()` mean calculate gradient

`optimizer.step()` mean update parameters

### Loss

Loss is the ways to measure how right or wrong prediction of model.

```python
loss_fn = loss_fn(output, labels) # measure the error of model
loss.backward() # calculate gradient or diagnose the problems
optimizer.step() # update parameters
```

List of Loss Function in PyTorch:

```python
nn.MSELoss() # Mean Squared Error for Regression
nn.CrossEntropyLoss() # Cross Entropy Loss for Classification
nn.BCELoss() # Binary Cross Entropy Loss for Binary Classification
nn.NLLLoss() # Negative Log Likelihood Loss for Classification
nn.KLDivLoss() # Kullback-Leibler Divergence Loss for Classification
nn.MarginRankingLoss() # Margin Ranking Loss for Classification
nn.SmoothL1Loss() # Smooth L1 Loss for Regression
nn.HingeEmbeddingLoss() # Hinge Embedding Loss for Classification
nn.MultiLabelMarginLoss() # Multi Label Margin Loss for Classification
nn.MultiMarginLoss() # Multi Margin Loss for Classification
nn.TripletMarginLoss() # Triplet Margin Loss for Classification
nn.CosineEmbeddingLoss() # Cosine Embedding Loss for Classification
nn.MultiLabelSoftMarginLoss() # Multi Label Soft Margin Loss for Classification
nn.SoftMarginLoss() # Soft Margin Loss for Classification
nn.HuberLoss() # Huber Loss for Regression
nn.PoissonNLLLoss() # Poisson Negative Log Likelihood Loss for Regression
nn.GaussianNLLLoss() # Gaussian Negative Log Likelihood Loss for Regression
nn.CTCLoss() # Connectionist Temporal Classification Loss for Sequence Classification
```

### Optimizer and Gradients

Optimizer is the ways to update parameters of model.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

list of optimizer in PyTorch:

```python
torch.optim.SGD() # Stochastic Gradient Descent
torch.optim.Adam() # Adaptive Momentum
torch.optim.RMSprop() # Root Mean Square Propagation
torch.optim.Adagrad() # Adaptive Gradient
torch.optim.Adadelta() # Adaptive Delta
torch.optim.AdamW() # Adaptive Weight
torch.optim.LBFGS() # Limited Memory Broyden–Fletcher–Goldfarb–Shanno
torch.optim.SparseAdam() # Sparse Adaptive Momentum
torch.optim.Rprop() # Resilient Propagation
```

## Evaluation

```python
with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')
```

`torch.no_grad()` mean disable gradient calculation

`torch.max(outputs.data, 1)` mean get the class or labels with highest score

`correct` mean correct prediction

`total` mean total prediction

`predicted` mean predicted label

`labels` mean true label

## Full Data Pipeline

```python
# define transform of data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])
# create dataset with transform
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# load the dataset with DataLoader
DataLoader(dataset, batch_size=64, shuffle=True)

# training models with data
for batch, (data, labels) in enumerate(dataloader):
    # model process batch of data transformed
    output = model(data)
```
