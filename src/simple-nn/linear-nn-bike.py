import torch
import torch.nn as nn
import torch.optim as optim
import helper.helper_utils as utils

# distances in km
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float32)
# times in minutes
times = torch.tensor([[6.96], [12.11], [16.77], [22.21], [26.52]], dtype=torch.float32)

# plot data
utils.plot_data(distances, times)

# model nn
model = nn.Sequential(nn.Linear(1, 1))

# define loss and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# training loop

for epoch in range(100):
    # 0 reset the optimizer
    optimizer.zero_grad()
    # 1. make predictions
    outputs = model(distances)
    # 2. calculate loss
    loss = loss_fn(outputs, times)
    # 3. calculate adjustments with backpropagation
    loss.backward()
    # 4. update weights of the models
    optimizer.step()

with torch.no_grad():
    test_distance = torch.tensor([[6.0]], dtype=torch.float32)
    predicted_time = model(test_distance)
    print(f"Predicted time for a distance of {test_distance.item()} km: {predicted_time.item()} minutes")

    if predicted_time.item() > 30:
        print("\nDecision: Do NOT take the job. You will likely be late.")
    else:
        print("\nDecision: Take the job. You will be on time.")


# Access the first (and only) layer in the sequential model
layer = model[0]

# Get weights and bias
weights = layer.weight.data.numpy()
bias = layer.bias.data.numpy()

print(f"Weight: {weights}")
print(f"Bias: {bias}")


# Combined dataset: bikes for short distances, cars for longer ones
new_distances = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]
], dtype=torch.float32)

# Corresponding delivery times in minutes
new_times = torch.tensor([
    [6.96], [9.67], [12.11], [14.56], [16.77], [21.7], [26.52], [32.47], [37.15], [42.35],
    [46.1], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.8], [73.88],
    [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
    [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]
], dtype=torch.float32)


# Normalize the new data
distances_mean = new_distances.mean()
distances_std = new_distances.std()
distances_norm = (new_distances - distances_mean) / distances_std

times_mean = new_times.mean()
times_std = new_times.std()
times_norm = (new_times - times_mean) / times_std

# Use the already-trained linear model to make predictions
with torch.no_grad():
    predictions = model(new_distances)

# Print the predictions
for i in range(len(new_distances)):
    print(f"Distance: {new_distances[i].item()} km -> Predicted Time: {predictions[i].item():.2f} minutes | Actual Time: {new_times[i].item():.2f} minutes | Error: {abs(predictions[i].item() - new_times[i].item()):.2f} minutes")

utils.plot_data(new_distances, new_times, normalize=False)
utils.plot_final_fit(model, new_distances, new_times, distances_norm, times_std, times_mean)