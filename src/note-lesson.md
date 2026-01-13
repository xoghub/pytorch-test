# Lesson Noted of PyTorch Specialization Course

The ML Pipeline

- Data Ingestion
  - raw data collection
- Data Preprocessing
  - data access
  - data quality
  - data efficiency
- Modeling
- Model Training
- Model Evaluation
- Model Deployment

## Data Ingestion

after raw data collection, we need to load data with batching (small chunk of data)

### Data Preprocessing

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

4. Full Data Pipeline

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
