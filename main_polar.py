from bspfunctions import SmartPSB
import numpy as np
import matplotlib.pyplot as plt
import random
from network import ConvNet, ConvVal
import torch
import warnings
from datasetgenerator import GridDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import time
warnings.filterwarnings('ignore')
torch.manual_seed(4)

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

n = 9  # N by N view of the robot, must be odd
path_planner = SmartPSB(num_y=n)
target = np.array([10, 2])
possible_actions = np.arange(0, n)
num_samples = 10000
batch_size = 10
griddataset = GridDataset(n, num_samples)
gridloader = DataLoader(griddataset, batch_size=batch_size, shuffle=True)
actor = ConvNet(grid_size=n).to(device)
critic = ConvVal(grid_size=n).to(device)

# Define parameters for the circle slice
theta1, theta2 = 0, 180  # Degrees
radius = 3
num_slices_radial = 10
num_slices_angular = 9
rotation_angle = -90

grid_centers = path_planner.calculate_grid_centers(radius, theta1, theta2, num_slices_radial, num_slices_angular, rotation_angle)
#####
# critic.load_state_dict(torch.load('ppo_critic.pth'))
# actor.load_state_dict(torch.load('ppo_actor.pth'))
####

learning_rate = 0.001
# Initialize optimizers for actor and critic
actor_optim = Adam(actor.parameters(), lr=learning_rate)
critic_optim = Adam(critic.parameters(), lr=learning_rate)
num_epochs = 3
n_updates_per_iteration = 5
ppo_clip = 0.2
save_frequency = 100
render = False

actor_losses = []
critic_losses = []
batch_rew_history = []
epsilon = 0.1  # Exploration rate

for epoch in range(num_epochs):
    for batch_idx, batch_grids in enumerate(gridloader):
        batch_grids = batch_grids.to(device)  # Move batch to GPU

        batch_actions = []
        batch_log_probs = []
        batch_rew = []
        ## rollout
        for i in range(batch_grids.shape[0]):
            grid = batch_grids[i]
            grid_tensor = torch.tensor(grid, dtype=torch.float).to(device)
            grid_tensor = grid_tensor.view(1, 1, n, n)

            gird_tensor_test = grid_tensor.view(n, n)
            gird_numpy_test = gird_tensor_test.detach().numpy()
            obs = path_planner.obstacle_from_grid_polar(gird_numpy_test)

            dist_map = actor(grid_tensor)
            dist_map_numpy = dist_map.detach().numpy()
            actions = []
            log_probs = []

            for i in range(n-1):
                action = random.choices(possible_actions, dist_map_numpy[0, :, i+1])[0]
                action_prob = dist_map_numpy[0, :, i+1][action]
                log_prob = np.log(action_prob)
                actions.append(action)
                log_probs.append(log_prob)

            log_probs = np.sum(log_probs)
            actions = np.array(actions)
            p = path_planner.action2point_polar(grid_centers, actions)
            path = path_planner.construct_sp(p)
            # plt.plot(path[:, 0], path[:, 1])
            # plt.show()
            cost = path_planner.calculate_cost(path, target, grid)
            batch_rew.append(cost)
            batch_actions.append(actions)
            batch_log_probs.append(log_probs)
        batch_rew = torch.tensor(batch_rew, dtype=torch.float32)
        batch_actions = torch.tensor(batch_actions, dtype=torch.int)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)
        batch_grids = batch_grids.view(batch_size, 1, n, n)
        batch_rew_mean = batch_rew.mean().item()
        batch_rew_history.append(batch_rew_mean)
        V = critic(batch_grids).squeeze()
        A_k = batch_rew - V.detach()
        # A_k = batch_rew - 1000
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
        for _ in range(n_updates_per_iteration):
            V = critic(batch_grids).squeeze()
            dist_map = actor(batch_grids)
            dist_map_numpy = dist_map.detach().numpy()
            curr_log_probs = []
            for ii in range(batch_actions.shape[0]):
                logss = 0
                for jj in range(batch_actions.shape[1]):
                    action_prob = dist_map[ii, :, jj+1][batch_actions[ii,jj]]
                    logss += torch.log(action_prob)
                curr_log_probs.append(logss)
            curr_log_probs = torch.stack(curr_log_probs)
            ratios = torch.exp(curr_log_probs - batch_log_probs)
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - ppo_clip, 1 + ppo_clip) * A_k
            actor_loss = (torch.min(surr1, surr2)).mean()
            if torch.isnan(actor_loss).any():
                print('Error: Actor loss is NaN')

            critic_loss = nn.MSELoss()(V, batch_rew)
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            print(f'Epoch: {epoch}/{num_epochs}, batch: {batch_idx}/{num_samples/batch_size}')
            print(f'Actor loss: {actor_loss.item():.3f}')
            print(f'Critic loss: {critic_loss.item():.3f}')
            # Calculate gradients and perform backward propagation for actor network
            actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optim.step()

            # Calculate gradients and perform backward propagation for critic network
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
            print('-'*10)


        if batch_idx == 0 and render:


            plt.plot(actor_losses)  # Example plot, replace with your actual plotting code
            plt.title("Actor loss")
            plt.show()
            plt.plot(critic_losses)  # Example plot, replace with your actual plotting code
            plt.title("Critic loss")
            plt.show()
            plt.plot(batch_rew_history)
            plt.title("Batch Rewards")
            plt.show()


            fig, axes = plt.subplots(2, 5, figsize=(20, 4))  # Adjust the size as needed
            axes = axes.flatten()  # Flatten the axes array for easy iteration
            success = []
            for idx, ax in enumerate(axes):
                grid_example_tensor = batch_grids[idx]
                grid_example_numpy = grid_example_tensor.detach().numpy().reshape(n, n)
                obstacles = path_planner.obstacle_from_grid(grid_example_numpy)
                dist_map = actor(grid_example_tensor)
                dist_map_numpy = dist_map.detach().numpy()

                # Plot grid
                ax.scatter(obstacles[:,0], obstacles[:,1], c='yellow', alpha=1)
                ax.imshow(np.concatenate((np.zeros((n, 1)), dist_map_numpy.reshape(n, n)), axis=1), cmap='gray', extent=[-0.5, n + 0.5, -n/2, n/2])

                ax.set_title(f"Grid {idx + 1}")

                actions = []
                log_probs = []
                for i in range(n-1):
                    action = random.choices(possible_actions, dist_map_numpy[0, :, i+1])[0]
                    action_prob = dist_map_numpy[0, :, i+1][action]
                    log_prob = np.log(action_prob)
                    actions.append(action)
                    log_probs.append(log_prob)

                actions = np.array(actions)
                print(f'Actions for grid {idx + 1} are {actions}')
                print('Grid: ', grid_example_numpy)
                print('Obstacles: ', obstacles)
                y = path_planner.action2point(actions)
                p = np.column_stack((x, y))  # Assuming x coordinates are sequential
                path = path_planner.construct_sp(p)
                collision = path_planner.obstacle_check(grid_example_numpy)
                print('Collision: ', collision)
                print('-'*10)
                # Plotting the path
                ax.plot(path[:, 0], path[:, 1], 'r-', label='Spline Path')  # Adjust path plotting as needed
                ax.plot(p[:, 0], p[:, 1], 'o-', label='Control Points')  # Plot control points
                ax.axis('equal')  # Ensure equal aspect ratio
                # ax.legend()

            plt.tight_layout()
            plt.show()
        if batch_idx % save_frequency == 0:
            # save the actor and critic network
            torch.save(actor.state_dict(), 'ppo_actor.pth')
            torch.save(critic.state_dict(), 'ppo_critic.pth')
print('Done!')
