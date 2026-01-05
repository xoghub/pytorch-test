import torch
import torch.nn as nn
import torch.optim as optim
import helper.helper_utils as utils

# Combined dataset: bikes for short distances, cars for longer ones
distances = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]
], dtype=torch.float32)

# Corresponding delivery times in minutes
times = torch.tensor([
    [6.96], [9.67], [12.11], [14.56], [16.77], [21.7], [26.52], [32.47], [37.15], [42.35],
    [46.1], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.8], [73.88],
    [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
    [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]
], dtype=torch.float32)

utils.plot_data(distances, times)

distances_mean = distances.mean()
distances_std = distances.std()

times_mean = times.mean()
times_std = times.std()

distances_norm = (distances - distances_mean) / distances_std
times_norm = (times - times_mean) / times_std

utils.plot_data(distances_norm, times_norm, normalize=True)

torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(1, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(distances_norm)
    loss = nn.MSELoss()(outputs, times_norm)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        utils.plot_training_progress(epoch, loss, model, distances_norm, times_norm)

print(f"Training completed.")
print(f"Final loss: {loss:.4f}")
utils.plot_final_fit(model, distances, times, distances_norm, times_std, times_mean)

distances_test = 21.0

with torch.no_grad():
    distances_tensor = torch.tensor([[distances_test]], dtype=torch.float32)
    new_distances_norm = (distances_tensor - distances_mean) / distances_std
    predicted_time_norm = model(new_distances_norm)
    predicted_time = (predicted_time_norm * times_std) + times_mean

    print(f"Predicted time for a distance of {distances_test} km: {predicted_time.item():.2f} minutes")

    if predicted_time.item() > 40:
        print("\nDecision: Do NOT take the job. You will likely be late.")
    else:
        if distances_test < 5:
            print(f"\nDecision: Yes, delivery is possible. Since the distance is {distances_test} km (<= 5 km), use a bike.")
        else:
            print(f"\nDecision: Yes, delivery is possible. Since the distance is {distances_test} km (> 5 km), use a car.")
