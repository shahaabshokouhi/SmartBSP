import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_width, screen_height = 800, 600

# Create the display surface
screen = pygame.display.set_mode((screen_width, screen_height))

# Set the title of the window
pygame.display.set_caption("2D Occupation Grid with Vehicle and Obstacles")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Grid dimensions
grid_size = 20  # Size of the squares
grid_width = screen_width // grid_size
grid_height = screen_height // grid_size

# Vehicle position
vehicle_pos = [grid_width // 2, grid_height // 2]
vehicle_direction = 'UP'  # Initial direction
car_size = grid_size * 2
# Load and scale the vehicle image
original_vehicle_image = pygame.image.load('vehicle.png')
vehicle_images = {
    'UP': pygame.transform.scale(original_vehicle_image, (car_size, car_size)),
    'DOWN': pygame.transform.rotate(pygame.transform.scale(original_vehicle_image, (car_size, car_size)), 180),
    'LEFT': pygame.transform.rotate(pygame.transform.scale(original_vehicle_image, (car_size, car_size)), 90),
    'RIGHT': pygame.transform.rotate(pygame.transform.scale(original_vehicle_image, (car_size, car_size)), -90),
}
# Generate obstacles
num_obstacles = 50  # Number of obstacles
obstacles = {(random.randint(0, grid_width-1), random.randint(0, grid_height-1)) for _ in range(num_obstacles)}

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Handle key presses
    keys = pygame.key.get_pressed()
    new_pos = list(vehicle_pos)
    if keys[pygame.K_LEFT]:
        new_pos[0] = max(vehicle_pos[0] - 1, 0)
        vehicle_direction = 'LEFT'
    if keys[pygame.K_RIGHT]:
        new_pos[0] = min(vehicle_pos[0] + 1, grid_width - 1)
        vehicle_direction = 'RIGHT'
    if keys[pygame.K_UP]:
        new_pos[1] = max(vehicle_pos[1] - 1, 0)
        vehicle_direction = 'UP'
    if keys[pygame.K_DOWN]:
        new_pos[1] = min(vehicle_pos[1] + 1, grid_height - 1)
        vehicle_direction = 'DOWN'

    # Update vehicle position if not moving into an obstacle
    if tuple(new_pos) not in obstacles:
        vehicle_pos = new_pos

    # Fill the screen with black
    screen.fill(WHITE)

    # Draw the grid and obstacles
    for x in range(0, screen_width, grid_size):
        for y in range(0, screen_height, grid_size):
            rect = pygame.Rect(x, y, grid_size, grid_size)
            if (x // grid_size, y // grid_size) in obstacles:
                pygame.draw.rect(screen, BLACK, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect, 1)

    # Draw the vehicle image
    screen.blit(vehicle_images[vehicle_direction], (vehicle_pos[0]*grid_size, vehicle_pos[1]*grid_size))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)

pygame.quit()
