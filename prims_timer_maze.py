import pygame
import random
import heapq
import pickle
from collections import deque
import time  # Import the time module

# Maze dimensions and cell size
WIDTH, HEIGHT = 1000, 1000
CELL_SIZE = 10
cols, rows = WIDTH // CELL_SIZE, HEIGHT // CELL_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Generator and Solver")
clock = pygame.time.Clock()

# Initialize the grid of cells
grid = []

def init_grid():
    global grid
    grid = []
    for y in range(rows):
        grid.append([])
        for x in range(cols):
            grid[y].append({
                'x': x,
                'y': y,
                'walls': [True, True, True, True],  # [Top, Right, Bottom, Left]
                'in_maze': False  # For Prim's algorithm
            })

def draw_cell(cell, color=(0, 0, 255), thickness=2):
    x = cell['x'] * CELL_SIZE
    y = cell['y'] * CELL_SIZE
    if cell['walls'][0]:
        pygame.draw.line(screen, color, (x, y), (x + CELL_SIZE, y), thickness)  # Top wall
    if cell['walls'][1]:
        pygame.draw.line(screen, color, (x + CELL_SIZE, y), (x + CELL_SIZE, y + CELL_SIZE), thickness)  # Right wall
    if cell['walls'][2]:
        pygame.draw.line(screen, color, (x + CELL_SIZE, y + CELL_SIZE), (x, y + CELL_SIZE), thickness)  # Bottom wall
    if cell['walls'][3]:
        pygame.draw.line(screen, color, (x, y + CELL_SIZE), (x, y), thickness)  # Left wall

def get_cell(x, y):
    if 0 <= x < cols and 0 <= y < rows:
        return grid[y][x]
    else:
        return None

def generate_maze():
    # Maze generation using Prim's Algorithm with animation
    init_grid()
    wall_list = []

    # Start with a random cell
    start_x = random.randint(0, cols - 1)
    start_y = random.randint(0, rows - 1)
    start_cell = grid[start_y][start_x]
    start_cell['in_maze'] = True

    # Add the walls of the starting cell to the wall list
    x, y = start_cell['x'], start_cell['y']
    walls = start_cell['walls']
    if y > 0:
        wall_list.append((x, y, 0))  # Top wall
    if x < cols - 1:
        wall_list.append((x, y, 1))  # Right wall
    if y < rows - 1:
        wall_list.append((x, y, 2))  # Bottom wall
    if x > 0:
        wall_list.append((x, y, 3))  # Left wall

    while wall_list:
        # Randomly select a wall from the list
        wx, wy, wall = random.choice(wall_list)
        wall_list.remove((wx, wy, wall))

        current_cell = grid[wy][wx]
        opposite_cell = None
        dx, dy = 0, 0

        if wall == 0:
            dy = -1
        elif wall == 1:
            dx = 1
        elif wall == 2:
            dy = 1
        elif wall == 3:
            dx = -1

        nx, ny = wx + dx, wy + dy
        opposite_cell = get_cell(nx, ny)

        if opposite_cell and not opposite_cell['in_maze']:
            # Remove the wall between current_cell and opposite_cell
            current_cell['walls'][wall] = False
            opposite_wall = (wall + 2) % 4  # Opposite wall index
            opposite_cell['walls'][opposite_wall] = False

            # Mark the opposite cell as part of the maze
            opposite_cell['in_maze'] = True

            # Add the walls of the opposite cell to the wall list
            ox, oy = opposite_cell['x'], opposite_cell['y']
            walls = opposite_cell['walls']
            if oy > 0 and walls[0]:
                wall_list.append((ox, oy, 0))  # Top wall
            if ox < cols - 1 and walls[1]:
                wall_list.append((ox, oy, 1))  # Right wall
            if oy < rows - 1 and walls[2]:
                wall_list.append((ox, oy, 2))  # Bottom wall
            if ox > 0 and walls[3]:
                wall_list.append((ox, oy, 3))  # Left wall

        # Visualization
        screen.fill((0, 0, 0))
        for row in grid:
            for cell in row:
                draw_cell(cell)

        # Highlight the current cell
        pygame.draw.rect(screen, (255, 0, 0), (current_cell['x'] * CELL_SIZE, current_cell['y'] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()
        clock.tick(60)

        # Handle events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    # Maze generation is complete
    # Save the maze as an image file
    pygame.image.save(screen, "maze.png")

def reset_visited():
    for row in grid:
        for cell in row:
            cell['visited'] = False
            cell.pop('parent', None)
            cell.pop('distance', None)
            cell.pop('g', None)
            cell.pop('h', None)
            cell.pop('f', None)
            # For maze generation
            cell['in_maze'] = False

def save_maze(filename):
    with open(filename, 'wb') as f:
        pickle.dump(grid, f)
    print(f"Maze saved to {filename}")

def load_maze(filename):
    global grid
    with open(filename, 'rb') as f:
        grid = pickle.load(f)
    print(f"Maze loaded from {filename}")

def random_mouse(start, end):
    start_time = time.time()  # Start time measurement

    current_cell = start
    path = [current_cell]

    # Create a Surface for visited cells
    visited_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Draw the maze walls
        screen.fill((0, 0, 0))

        # Draw visited cells
        screen.blit(visited_surface, (0, 0))

        # Draw the head (current position) in white
        x = current_cell['x'] * CELL_SIZE
        y = current_cell['y'] * CELL_SIZE
        pygame.draw.rect(screen, (255, 255, 255), (x, y, CELL_SIZE, CELL_SIZE))

        # Draw the walls on top
        for row in grid:
            for cell_draw in row:
                draw_cell(cell_draw)

        pygame.display.flip()
        clock.tick(60)

        if current_cell == end:
            running = False
            continue

        neighbors = []
        walls = current_cell['walls']
        x, y = current_cell['x'], current_cell['y']

        if not walls[0] and y > 0:
            neighbors.append(grid[y - 1][x])  # Up
        if not walls[1] and x < cols - 1:
            neighbors.append(grid[y][x + 1])  # Right
        if not walls[2] and y < rows - 1:
            neighbors.append(grid[y + 1][x])  # Down
        if not walls[3] and x > 0:
            neighbors.append(grid[y][x - 1])  # Left

        if neighbors:
            current_cell = random.choice(neighbors)
            path.append(current_cell)
            # Draw the visited cell onto the visited_surface
            x = current_cell['x'] * CELL_SIZE
            y = current_cell['y'] * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(visited_surface, (34, 139, 34, 100), rect)
        else:
            if len(path) > 1:
                path.pop()
                current_cell = path[-1]
            else:
                # No path found
                running = False

    # Draw the final path with full-opacity dark green
    for cell in path:
        x = cell['x'] * CELL_SIZE
        y = cell['y'] * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (34, 139, 34), rect)

    # Draw the walls on top
    for row in grid:
        for cell_draw in row:
            draw_cell(cell_draw)

    pygame.display.flip()
    pygame.time.wait(500)

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Random Mouse Algorithm completed in {elapsed_time:.4f} seconds")

def flood_fill(start, end):
    start_time = time.time()  # Start time measurement

    queue = deque()
    start['visited'] = True
    queue.append(start)

    # Create a Surface for visited cells
    visited_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    running = True
    found_end = False
    while running and queue:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Draw the maze walls
        screen.fill((0, 0, 0))

        # Draw visited cells
        screen.blit(visited_surface, (0, 0))

        # Draw the walls on top
        for row in grid:
            for cell_draw in row:
                draw_cell(cell_draw)

        pygame.display.flip()
        clock.tick(60)

        current_cell = queue.popleft()

        # Draw the current visited cell onto the visited_surface
        x = current_cell['x'] * CELL_SIZE
        y = current_cell['y'] * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(visited_surface, (34, 139, 34, 100), rect)

        if current_cell == end:
            found_end = True
            break

        neighbors = []
        walls = current_cell['walls']
        x, y = current_cell['x'], current_cell['y']

        if not walls[0] and y > 0:
            neighbors.append(grid[y - 1][x])  # Up
        if not walls[1] and x < cols - 1:
            neighbors.append(grid[y][x + 1])  # Right
        if not walls[2] and y < rows - 1:
            neighbors.append(grid[y + 1][x])  # Down
        if not walls[3] and x > 0:
            neighbors.append(grid[y][x - 1])  # Left

        for neighbor in neighbors:
            if not neighbor['visited']:
                neighbor['visited'] = True
                neighbor['parent'] = current_cell
                queue.append(neighbor)

    if found_end:
        # Reconstruct path
        path = []
        current = end
        while current != start:
            path.append(current)
            current = current['parent']
        path.append(start)
        path.reverse()

        # Draw the final path with full-opacity dark green
        for cell in path:
            x = cell['x'] * CELL_SIZE
            y = cell['y'] * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (34, 139, 34), rect)

        # Draw the walls on top
        for row in grid:
            for cell_draw in row:
                draw_cell(cell_draw)

        pygame.display.flip()
        pygame.time.wait(500)

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Flood Fill Algorithm completed in {elapsed_time:.4f} seconds")

def left_hand(start, end):
    start_time = time.time()  # Start time measurement

    current_cell = start
    direction = 'E'  # Starting direction
    path = [current_cell]

    # Create a Surface for visited cells
    visited_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # Map for turning left
    left_turn = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}
    # Map for moving forward
    move_forward = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
    # Map for checking walls
    wall_index = {'N': 0, 'E': 1, 'S': 2, 'W': 3}

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Draw maze walls
        screen.fill((0, 0, 0))
        # Blit visited cells Surface
        screen.blit(visited_surface, (0, 0))

        # Draw the head (current position) in white
        x = current_cell['x'] * CELL_SIZE
        y = current_cell['y'] * CELL_SIZE
        pygame.draw.rect(screen, (255, 255, 255), (x, y, CELL_SIZE, CELL_SIZE))

        # Draw the walls on top
        for row in grid:
            for cell_draw in row:
                draw_cell(cell_draw)

        pygame.display.flip()
        clock.tick(60)

        if current_cell == end:
            running = False
            continue

        # Draw the current visited cell onto the visited_surface
        x = current_cell['x'] * CELL_SIZE
        y = current_cell['y'] * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(visited_surface, (34, 139, 34, 100), rect)

        # Turn left
        direction = left_turn[direction]

        # Check if left cell is open
        walls = current_cell['walls']
        x, y = current_cell['x'], current_cell['y']

        for _ in range(4):
            idx = wall_index[direction]
            if not walls[idx]:
                dx, dy = move_forward[direction]
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    current_cell = grid[ny][nx]
                    path.append(current_cell)
                    break
            else:
                # Turn right if blocked
                direction = left_turn[left_turn[left_turn[direction]]]

    # Draw the final path with full-opacity dark green
    for cell in path:
        x = cell['x'] * CELL_SIZE
        y = cell['y'] * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (34, 139, 34), rect)

    # Draw the walls on top
    for row in grid:
        for cell_draw in row:
            draw_cell(cell_draw)

    pygame.display.flip()
    pygame.time.wait(500)

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Left-hand Following Algorithm completed in {elapsed_time:.4f} seconds")

def right_hand(start, end):
    start_time = time.time()  # Start time measurement

    current_cell = start
    direction = 'E'  # Starting direction
    path = [current_cell]

    # Create a Surface for visited cells
    visited_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # Map for turning right
    right_turn = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
    # Map for moving forward
    move_forward = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
    # Map for checking walls
    wall_index = {'N': 0, 'E': 1, 'S': 2, 'W': 3}

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Draw maze walls
        screen.fill((0, 0, 0))
        # Blit visited cells Surface
        screen.blit(visited_surface, (0, 0))

        # Draw the head (current position) in white
        x = current_cell['x'] * CELL_SIZE
        y = current_cell['y'] * CELL_SIZE
        pygame.draw.rect(screen, (255, 255, 255), (x, y, CELL_SIZE, CELL_SIZE))

        # Draw the walls on top
        for row in grid:
            for cell_draw in row:
                draw_cell(cell_draw)

        pygame.display.flip()
        clock.tick(60)

        if current_cell == end:
            running = False
            continue

        # Draw the current visited cell onto the visited_surface
        x = current_cell['x'] * CELL_SIZE
        y = current_cell['y'] * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(visited_surface, (34, 139, 34, 100), rect)

        # Turn right
        direction = right_turn[direction]

        # Check if right cell is open
        walls = current_cell['walls']
        x, y = current_cell['x'], current_cell['y']

        for _ in range(4):
            idx = wall_index[direction]
            if not walls[idx]:
                dx, dy = move_forward[direction]
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    current_cell = grid[ny][nx]
                    path.append(current_cell)
                    break
            else:
                # Turn left if blocked
                direction = right_turn[right_turn[right_turn[direction]]]

    # Draw the final path with full-opacity dark green
    for cell in path:
        x = cell['x'] * CELL_SIZE
        y = cell['y'] * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (34, 139, 34), rect)

    # Draw the walls on top
    for row in grid:
        for cell_draw in row:
            draw_cell(cell_draw)

    pygame.display.flip()
    pygame.time.wait(500)

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Right-hand Following Algorithm completed in {elapsed_time:.4f} seconds")

def dijkstra(start, end):
    start_time = time.time()  # Start time measurement

    queue = []
    start['distance'] = 0
    heapq.heappush(queue, (0, start['x'], start['y'], start))

    # Create a Surface for visited cells
    visited_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    running = True
    found_end = False
    while running and queue:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Draw maze walls
        screen.fill((0, 0, 0))
        # Blit visited cells Surface
        screen.blit(visited_surface, (0, 0))
        # Draw the walls on top
        for row in grid:
            for cell_draw in row:
                draw_cell(cell_draw)

        pygame.display.flip()
        clock.tick(60)

        dist, _, _, current_cell = heapq.heappop(queue)
        if 'visited' in current_cell and current_cell['visited']:
            continue
        current_cell['visited'] = True

        # Draw the current visited cell onto the visited_surface
        x = current_cell['x'] * CELL_SIZE
        y = current_cell['y'] * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(visited_surface, (34, 139, 34, 100), rect)

        if current_cell == end:
            found_end = True
            break

        neighbors = []
        walls = current_cell['walls']
        x, y = current_cell['x'], current_cell['y']

        if not walls[0] and y > 0:
            neighbors.append(grid[y - 1][x])  # Up
        if not walls[1] and x < cols - 1:
            neighbors.append(grid[y][x + 1])  # Right
        if not walls[2] and y < rows - 1:
            neighbors.append(grid[y + 1][x])  # Down
        if not walls[3] and x > 0:
            neighbors.append(grid[y][x - 1])  # Left

        for neighbor in neighbors:
            alt = dist + 1
            if 'distance' not in neighbor or alt < neighbor['distance']:
                neighbor['distance'] = alt
                neighbor['parent'] = current_cell
                heapq.heappush(queue, (alt, neighbor['x'], neighbor['y'], neighbor))

    if found_end:
        # Reconstruct path
        path = []
        current = end
        while current != start:
            path.append(current)
            current = current['parent']
        path.append(start)
        path.reverse()

        # Draw the final path with full-opacity dark green
        for cell in path:
            x = cell['x'] * CELL_SIZE
            y = cell['y'] * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (34, 139, 34), rect)

        # Draw the walls on top
        for row in grid:
            for cell_draw in row:
                draw_cell(cell_draw)

        pygame.display.flip()
        pygame.time.wait(500)

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Dijkstra's Algorithm completed in {elapsed_time:.4f} seconds")

def a_star(start, end):
    start_time = time.time()  # Start time measurement

    open_set = []
    start['g'] = 0
    start['h'] = abs(end['x'] - start['x']) + abs(end['y'] - start['y'])
    start['f'] = start['g'] + start['h']
    heapq.heappush(open_set, (start['f'], start['h'], start['x'], start['y'], start))

    # Create a Surface for visited cells
    visited_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    running = True
    found_end = False
    while running and open_set:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Draw maze walls
        screen.fill((0, 0, 0))
        # Blit visited cells Surface
        screen.blit(visited_surface, (0, 0))
        # Draw the walls on top
        for row in grid:
            for cell_draw in row:
                draw_cell(cell_draw)

        pygame.display.flip()
        clock.tick(60)

        _, _, _, _, current_cell = heapq.heappop(open_set)
        if 'visited' in current_cell and current_cell['visited']:
            continue
        current_cell['visited'] = True

        # Draw the current visited cell onto the visited_surface
        x = current_cell['x'] * CELL_SIZE
        y = current_cell['y'] * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(visited_surface, (34, 139, 34, 100), rect)

        if current_cell == end:
            found_end = True
            break

        neighbors = []
        walls = current_cell['walls']
        x, y = current_cell['x'], current_cell['y']

        if not walls[0] and y > 0:
            neighbors.append(grid[y - 1][x])  # Up
        if not walls[1] and x < cols - 1:
            neighbors.append(grid[y][x + 1])  # Right
        if not walls[2] and y < rows - 1:
            neighbors.append(grid[y + 1][x])  # Down
        if not walls[3] and x > 0:
            neighbors.append(grid[y][x - 1])  # Left

        for neighbor in neighbors:
            tentative_g = current_cell['g'] + 1
            if 'g' not in neighbor or tentative_g < neighbor['g']:
                neighbor['g'] = tentative_g
                neighbor['h'] = abs(end['x'] - neighbor['x']) + abs(end['y'] - neighbor['y'])
                neighbor['f'] = neighbor['g'] + neighbor['h']
                neighbor['parent'] = current_cell
                heapq.heappush(open_set, (neighbor['f'], neighbor['h'], neighbor['x'], neighbor['y'], neighbor))

    if found_end:
        # Reconstruct path
        path = []
        current = end
        while current != start:
            path.append(current)
            current = current['parent']
        path.append(start)
        path.reverse()

        # Draw the final path with full-opacity dark green
        for cell in path:
            x = cell['x'] * CELL_SIZE
            y = cell['y'] * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (34, 139, 34), rect)

        # Draw the walls on top
        for row in grid:
            for cell_draw in row:
                draw_cell(cell_draw)

        pygame.display.flip()
        pygame.time.wait(500)

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"A* Pathfinding Algorithm completed in {elapsed_time:.4f} seconds")

def dead_end_filling(start, end):
    start_time = time.time()  # Start time measurement

    # Create a Surface for the visualization
    visited_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # Create a set to keep track of filled cells
    filled_cells = set()

    # Function to check if a cell is a dead-end
    def is_dead_end(cell):
        if cell == start or cell == end:
            return False  # Don't consider start or end
        openings = 0
        x, y = cell['x'], cell['y']
        walls = cell['walls']
        directions = [(0, -1, 0),  # Up
                      (1, 0, 1),   # Right
                      (0, 1, 2),   # Down
                      (-1, 0, 3)]  # Left
        for dx, dy, idx in directions:
            if not walls[idx]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    neighbor = grid[ny][nx]
                    if (nx, ny) not in filled_cells:
                        openings += 1
        return openings == 1

    # Initialize the list of dead-end cells
    dead_ends = []

    # Find initial dead-ends
    for row in grid:
        for cell in row:
            if is_dead_end(cell):
                dead_ends.append(cell)

    running = True
    while running:
        if not dead_ends:
            running = False
            continue

        new_dead_ends = []
        for cell in dead_ends:
            x, y = cell['x'], cell['y']
            while is_dead_end(cell):
                filled_cells.add((cell['x'], cell['y']))

                # Draw the current cell onto the visited_surface
                rect = pygame.Rect(cell['x'] * CELL_SIZE, cell['y'] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(visited_surface, (34, 139, 34, 100), rect)

                # Update the display
                # Draw the maze walls
                screen.fill((0, 0, 0))

                for row in grid:
                    for cell_draw in row:
                        draw_cell(cell_draw)

                # Blit the highlighted dead-ends
                screen.blit(visited_surface, (0, 0))

                # Draw the walls on top
                for row in grid:
                    for cell_draw in row:
                        draw_cell(cell_draw)

                pygame.display.flip()
                clock.tick(60)

                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()

                # Move to the next cell
                moved = False
                walls = cell['walls']
                directions = [(0, -1, 0),  # Up
                              (1, 0, 1),   # Right
                              (0, 1, 2),   # Down
                              (-1, 0, 3)]  # Left
                for dx, dy, idx in directions:
                    if not walls[idx]:
                        nx, ny = cell['x'] + dx, cell['y'] + dy
                        if 0 <= nx < cols and 0 <= ny < rows:
                            neighbor = grid[ny][nx]
                            if (nx, ny) not in filled_cells:
                                cell = neighbor
                                moved = True
                                break
                if not moved:
                    break

            # After filling this path, check neighboring cells for new dead-ends
            for dx, dy, idx in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    neighbor = grid[ny][nx]
                    if is_dead_end(neighbor) and (neighbor['x'], neighbor['y']) not in filled_cells:
                        if neighbor not in new_dead_ends:
                            new_dead_ends.append(neighbor)

        dead_ends = new_dead_ends

    # Draw the maze walls one final time
    screen.fill((0, 0, 0))
    for row in grid:
        for cell_draw in row:
            draw_cell(cell_draw)
    screen.blit(visited_surface, (0, 0))
    for row in grid:
        for cell_draw in row:
            draw_cell(cell_draw)
    pygame.display.flip()

    # Find the solution path using DFS
    reset_visited()
    path = []

    def dfs(cell):
        if cell == end:
            path.append(cell)
            return True
        cell['visited'] = True
        x, y = cell['x'], cell['y']
        walls = cell['walls']

        directions = [(0, -1, 0),  # Up
                      (1, 0, 1),   # Right
                      (0, 1, 2),   # Down
                      (-1, 0, 3)]  # Left
        for dx, dy, idx in directions:
            if not walls[idx]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    neighbor = grid[ny][nx]
                    if not neighbor['visited'] and (neighbor['x'], neighbor['y']) not in filled_cells:
                        if dfs(neighbor):
                            path.append(cell)
                            return True
        return False

    dfs(start)
    path.reverse()

    # Visualize the solution path
    for cell in path:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Draw the maze walls
        screen.fill((0, 0, 0))

        for row in grid:
            for cell_draw in row:
                draw_cell(cell_draw)

        # Blit the highlighted dead-ends
        screen.blit(visited_surface, (0, 0))

        # Draw the solution path
        for cell_in_path in path:
            x = cell_in_path['x'] * CELL_SIZE
            y = cell_in_path['y'] * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (34, 139, 34), rect)

        # Draw the walls on top
        for row in grid:
            for cell_draw in row:
                draw_cell(cell_draw)

        pygame.display.flip()
        clock.tick(60)

    # Pause to show the final path
    pygame.time.wait(500)
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"Dead-end Filling Algorithm completed in {elapsed_time:.4f} seconds")

# Main function
def main():
    init_grid()

    print("Welcome to the Maze Generator and Solver!")
    print("1. Generate a new maze using Prim's Algorithm")
    print("2. Load an existing maze")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        generate_maze()
        save_option = input("Do you want to save the generated maze? (y/n): ")
        if save_option.lower() == 'y':
            filename = input("Enter filename to save the maze (e.g., maze_data.pkl): ")
            save_maze(filename)
    elif choice == '2':
        filename = input("Enter filename of the maze to load (e.g., maze_data.pkl): ")
        try:
            load_maze(filename)
            # Redraw the loaded maze
            screen.fill((0, 0, 0))
            for row in grid:
                for cell in row:
                    draw_cell(cell)
            pygame.display.flip()
        except FileNotFoundError:
            print("File not found. Exiting.")
            pygame.quit()
            exit()
    else:
        print("Invalid choice. Exiting.")
        pygame.quit()
        exit()

    while True:
        reset_visited()

        # Starting and ending cells
        start = grid[0][0]
        end = grid[rows - 1][cols - 1]

        print("Select a maze-solving algorithm:")
        print("1. Random Mouse")
        print("2. Flood Fill")
        print("3. Left-hand Following")
        print("4. Right-hand Following")
        print("5. Dijkstra's Algorithm")
        print("6. A* Pathfinding Algorithm")
        print("7. Dead-end Filling Algorithm")
        choice = input("Enter the number of your choice: ")

        if choice == '1':
            random_mouse(start, end)
        elif choice == '2':
            flood_fill(start, end)
        elif choice == '3':
            left_hand(start, end)
        elif choice == '4':
            right_hand(start, end)
        elif choice == '5':
            dijkstra(start, end)
        elif choice == '6':
            a_star(start, end)
        elif choice == '7':
            dead_end_filling(start, end)
        else:
            print("Invalid choice.")

        again = input("Do you want to try another algorithm on the same maze? (y/n): ")
        if again.lower() != 'y':
            break

    # Keep the window open until closed by the user
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(30)
    pygame.quit()

if __name__ == "__main__":
    main()
