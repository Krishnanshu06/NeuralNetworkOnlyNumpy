import pygame
import numpy as np

# Initialize pygame
pygame.init()

# Constants
GRID_SIZE = 28  
CELL_SIZE = 20  
SIDEBAR_WIDTH = 100  
DISPLAY_WIDTH = 100  
WIDTH, HEIGHT = GRID_SIZE * CELL_SIZE + SIDEBAR_WIDTH + DISPLAY_WIDTH, GRID_SIZE * CELL_SIZE + 50  
WHITE, BLACK, GRAY, BLUE, RED, DARK_GRAY = (255, 255, 255), (0, 0, 0), (200, 200, 200), (0, 150, 255), (200, 50, 50), (100, 100, 100)

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("28x28 Drawing Grid with Prediction Display")

# Grid representation (28x28, all initially 0)
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

# Brush settings
pen_mode = True  
brush_size = 1  
predicted_number = None  # Holds the predicted digit

# Button properties
BUTTON_PREDICT = pygame.Rect(WIDTH - SIDEBAR_WIDTH + 10, HEIGHT - 90, 80, 30)  
BUTTON_CLEAR = pygame.Rect(WIDTH - SIDEBAR_WIDTH + 10, HEIGHT - 50, 80, 30)  

# Import your CheckDigit function
from main import CheckDigit  # Replace with actual file name

# Function to draw the grid
def draw_grid():
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * CELL_SIZE + DISPLAY_WIDTH, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = WHITE if grid[y, x] == 0 else BLACK  
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)  

# Function to draw the brush size slider and prediction area
def draw_sidebar():
    pygame.draw.rect(screen, DARK_GRAY, (0, 0, DISPLAY_WIDTH, HEIGHT))  
    font = pygame.font.Font(None, 24)

    text = font.render("Size: " + str(brush_size), True, WHITE)
    screen.blit(text, (WIDTH - SIDEBAR_WIDTH + 20, 10))
    
    pygame.draw.rect(screen, WHITE, (WIDTH - SIDEBAR_WIDTH + 20, 40, 60, 10))
    
    knob_x = WIDTH - SIDEBAR_WIDTH + 20 + int((brush_size - 1) / 9 * 60)
    pygame.draw.circle(screen, RED, (knob_x, 45), 5)

    pygame.draw.rect(screen, BLUE, BUTTON_PREDICT)
    pygame.draw.rect(screen, RED, BUTTON_CLEAR)

    text_predict = font.render("Predict", True, WHITE)
    text_clear = font.render("Clear", True, WHITE)
    screen.blit(text_predict, (BUTTON_PREDICT.x + 15, BUTTON_PREDICT.y + 5))
    screen.blit(text_clear, (BUTTON_CLEAR.x + 25, BUTTON_CLEAR.y + 5))

    # Draw prediction display area
    pygame.draw.rect(screen, BLACK, (10, HEIGHT // 2 - 50, 80, 100), 3)
    pred_text = font.render("Prediction:", True, BLACK)
    screen.blit(pred_text, (15, HEIGHT // 2 - 70))

    if predicted_number is not None:
        num_text = font.render(str(predicted_number), True, BLACK)
        screen.blit(num_text, (45, HEIGHT // 2 - 20))

# Function to draw the pen/eraser
def draw_brush(x, y):
    global grid
    radius = brush_size // 2  

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE:
                if pen_mode:
                    grid[new_y, new_x] = 255  
                else:
                    grid[new_y, new_x] = 0  

# Function to process prediction (calls CheckDigit)
def predict_digit():
    global predicted_number  
    normalized_array = grid.astype(np.float32) / 255.0  # Convert values to range [0,1]
    flattened = normalized_array.flatten().reshape(1, 784)  # Convert to (1, 784)

    # Call CheckDigit function and store the returned prediction
    predicted_number = CheckDigit(flattened)
    print(f"Predicted: {predicted_number}")

# Function to clear the grid and reset prediction
def clear_grid():
    global grid, predicted_number
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)  
    predicted_number = None  
    print("Grid cleared!")

# Main loop
running = True
while running:
    screen.fill(WHITE)

    draw_grid()
    draw_sidebar()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if pygame.mouse.get_pressed()[0]:  
            x, y = pygame.mouse.get_pos()

            if WIDTH - SIDEBAR_WIDTH + 20 <= x <= WIDTH - SIDEBAR_WIDTH + 80 and 40 <= y <= 50:
                brush_size = 1 + int((x - (WIDTH - SIDEBAR_WIDTH + 20)) / 60 * 9)  

            elif DISPLAY_WIDTH <= x < GRID_SIZE * CELL_SIZE + DISPLAY_WIDTH:  
                grid_x, grid_y = (x - DISPLAY_WIDTH) // CELL_SIZE, y // CELL_SIZE
                draw_brush(grid_x, grid_y)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                pen_mode = True
            elif event.key == pygame.K_e:
                pen_mode = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if BUTTON_PREDICT.collidepoint(event.pos):
                predict_digit()  # Call CheckDigit() and display result
            if BUTTON_CLEAR.collidepoint(event.pos):
                clear_grid()  

    pygame.display.flip()

pygame.quit()
