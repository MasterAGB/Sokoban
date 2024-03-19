import json
import random
import pygame

class SolsticeGame:
    def __init__(self, level_index=1, game_skin ="default"):
        self.level_name = None
        self.skins = ['default', 'portal', 'bombs', 'forest', 'ice', 'castle']
        self.skin = game_skin
        self.last_action = None
        global win, win_size;
        self.LoadLevel(level_index)

        self.is_dizzy = False
        self.state = 0
        self.done = False
        # Define action mapping: 0=Left, 1=Down, 2=Right, 3=Up
        # self.action_space = np.arange(4)
        # self.observation_space = np.arange(len(self.map_layout) * len(self.map_layout[0]))
        self.enableRendering = True

        self.level_size = len(self.map_layout) * len(self.map_layout[0])
        self.action_size = 4  # Assuming Left, Down, Right, Up

        pygame.init()
        win_size = (737, 744)
        win = pygame.display.set_mode(win_size)
        pygame.display.set_caption("Solstice Play")

    def action_space_sample(self):
        # Returns a random action from 0, 1, 2, 3
        return random.randint(0, 3)

    def generate_solvable_map(self, rows, cols):
        # Initialize map with 'F' (free) cells
        game_map = [['.' for _ in range(cols)] for _ in range(rows)]

        # Starting point
        game_map[0][0] = 'S'

        # Ensure 'G' is placed at the bottom-right corner
        game_map[-1][-1] = 'G'

        # Create a guaranteed solvable path from S to G
        cur_row, cur_col = 0, 0

        # Randomly add holes, ensuring 'G' remains at its location
        for _ in range(int(rows * cols * 0.6)):  # Adjust density of 'H' as needed
            r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
            if game_map[r][c] == '.':
                game_map[r][c] = 'H'

        while cur_row < rows - 1 or cur_col < cols - 1:
            if cur_row < rows - 1 and (random.choice([True, False]) or cur_col == cols - 1):
                cur_row += 1  # Move down
            else:
                cur_col += 1  # Move right
            if cur_row < rows and cur_col < cols:
                game_map[cur_row][cur_col] = 'U'  # Mark path as 'F'

        # Reaffirm 'G' placement in case of overwrites
        game_map[-1][-1] = 'G'

        return ["".join(row) for row in game_map]

    def load_map(self, level_index):

        # Construct the file name based on the level identifier
        file_name = f"levels/level_{level_index}.json"

        try:
            # Open the level file and load its content
            with open(file_name, 'r') as file:
                level = json.load(file)
                self.level_name = level['name']  # Store level name
                self.skin = level['style']  # Set skin based on level
                return level['map_structure']
        except FileNotFoundError:
            print(f"Level file {file_name} not found - WIN the game.")
            return self.generate_solvable_map(8, 8)  # Fallback to a default map
        except json.JSONDecodeError:
            print(f"Error reading level file {file_name}.")
            return self.generate_solvable_map(8, 8)  # Fallback to a default map

    def step(self, action):

        if self.is_dizzy:
            # With some probability, choose a random action instead of the intended one
            if random.random() < 0.33:  # Adjust this probability as needed
                action = self.action_space_sample()

        # Simplified step function with directional movement.
        rows = len(self.map_layout)
        cols = len(self.map_layout[0])
        reward = 0
        terminated = False
        truncated = False  # Assuming this version doesn't use truncation.
        info = {}  # Placeholder for additional info.

        # Calculate current position
        row = self.state // cols
        col = self.state % cols
        row_prev = row;
        col_prev = col;

        self.last_action = action;
        # Determine new position based on action
        if action == 0:  # Left
            col = max(0, col - 1)
        elif action == 1:  # Down
            row = min(rows - 1, row + 1)
        elif action == 2:  # Right
            col = min(cols - 1, col + 1)
        elif action == 3:  # Up
            row = max(0, row - 1)

        # Update state
        self.state = row * cols + col

        self.moveAllMobs("M", ".")

        # Check for game termination conditions
        cell_prev= self.map_layout[row_prev][col_prev]
        if cell_prev == 'U':
            self.replaceThisCell(row_prev, col_prev, "H")

        cell = self.map_layout[row][col]
        if cell == 'G':
            terminated = True
            reward = 1
        elif cell == 'H':
            terminated = True
        elif cell == 'M':
            terminated = True
        elif cell == 'D':
            self.is_dizzy = True
            self.replaceThisCell(row, col, ".")
        elif cell == 'P':
            self.is_dizzy = False
            self.replaceThisCell(row, col, ".")
        elif cell == 'K':
            reward = 0.5
            self.replaceThisCell(row, col, ".")
            self.replaceAllCells("C", "G")


        self.render();

        return self.state, reward, terminated, truncated, info

    def replaceThisCell(self, row, col, new_type):
        """
        Replace the cell at the specified row and column with the new type.
        """
        if 0 <= row < len(self.map_layout) and 0 <= col < len(self.map_layout[0]):
            self.map_layout[row] = self.map_layout[row][:col] + new_type + self.map_layout[row][col + 1:]

    def moveAllMobs(self, mob_type, allowed_tile):
        rows = len(self.map_layout)
        cols = len(self.map_layout[0])
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
        mobs_positions = [(r, c) for r in range(rows) for c in range(cols) if self.map_layout[r][c] == mob_type]

        for r, c in mobs_positions:
            random.shuffle(directions)  # Shuffle directions to randomize mob movement
            moved = False
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < rows and 0 <= new_c < cols and self.map_layout[new_r][new_c] == allowed_tile:
                    # Move mob to new position
                    self.map_layout[r] = self.map_layout[r][:c] + allowed_tile + self.map_layout[r][c + 1:]
                    self.map_layout[new_r] = self.map_layout[new_r][:new_c] + mob_type + self.map_layout[new_r][
                                                                                         new_c + 1:]
                    moved = True
                    break  # Break after moving to avoid trying other directions

            if not moved:
                # If the mob cannot move (all adjacent tiles are not allowed), it stays in its current position
                continue

    def replaceAllCells(self, from_type, to_type):
        """
        Replace all instances of one cell type with another throughout the map.
        """
        for row in range(len(self.map_layout)):
            self.map_layout[row] = self.map_layout[row].replace(from_type, to_type)

    def GetDefaultPlayerPosition(self):
        """
        Finds the Start tile ('S') in the map layout and calculates the state for it.

        Returns:
            int: The state corresponding to the Start tile's position.
        """
        for row_index, row in enumerate(self.map_layout):
            if 'S' in row:
                col_index = row.index('S')
                return row_index * len(self.map_layout[0]) + col_index
        # Fallback in case 'S' is not found, though this should not happen
        return 0

    def reset(self):
        self.LoadLevel(self.level_index);
        self.state = self.GetDefaultPlayerPosition()
        self.done = False
        self.is_dizzy = False

        self.render();

        return self.state, {}

    def render(self):
        global win, win_size
        if self.enableRendering == False:
            return

        def load_image(skin, name):
            """Loads an image from the 'tiles/' directory."""
            return pygame.image.load(f'tiles/{skin}/{name}.png')

        scale_factor = 2;

        def load_image_scaled(skin, image_name, scale_factor=2):
            # Load the image using your existing load_image function
            image = load_image(skin, image_name)
            # Get the current size of the image
            original_size = image.get_size()
            # Calculate the new size based on the scale factor
            new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
            # Scale the image to the new size
            scaled_image = pygame.transform.scale(image, new_size)
            return scaled_image


        # Load tiles with added default and wall tiles
        tiles = {
            'CL': load_image_scaled(('char_dizzy' if self.is_dizzy else 'char'), 'char_left', scale_factor),
            'CB': load_image_scaled(('char_dizzy' if self.is_dizzy else 'char'), 'char_bottom', scale_factor),
            'CR': load_image_scaled(('char_dizzy' if self.is_dizzy else 'char'), 'char_right', scale_factor),
            'CT': load_image_scaled(('char_dizzy' if self.is_dizzy else 'char'), 'char_top', scale_factor),
            'S': load_image_scaled(self.skin, 'start', scale_factor),
            '.': load_image_scaled(self.skin, 'free', scale_factor),
            'H': load_image_scaled(self.skin, 'hole', scale_factor),
            'G': load_image_scaled(self.skin, 'goal', scale_factor),
            'C': load_image_scaled(self.skin, 'goalClosed', scale_factor),
            'K': load_image_scaled(self.skin, 'key', scale_factor),
            'M': load_image_scaled(self.skin, 'mob', scale_factor),
            'U': load_image_scaled(self.skin, 'unstable', scale_factor),
            'D': load_image_scaled(self.skin, 'dizzy', scale_factor),
            'P': load_image_scaled(self.skin, 'potion', scale_factor),
            'WTL': load_image_scaled(self.skin, 'wall', scale_factor),  # Wall Top Left
            'WTR': load_image_scaled(self.skin, 'wall', scale_factor),  # Wall Top Right
        }

        state = self.state
        map_layout = self.map_layout
        grid_size_rows = len(self.map_layout)
        grid_size_cols = len(self.map_layout[0])
        # Assume each tile has fixed size for simplicity
        tile_width, tile_height = 32 * scale_factor, 16 * scale_factor

        # This time, calculate offset to center the (-1, -1) tile
        # First, find the center of the window
        center_x = win_size[0] / 2
        center_y = win_size[1] / 2

        # Calculate offset_x such that (-1, -1) tile's center is at window's center
        # We adjust for half a tile width because (-1, -1) is off-center to the left
        # and we move it up by half the total map height in isometric projection to align top
        offset_x = center_x - (tile_width // 2)

        # Calculate offset_y to place (-1, -1) at the top of the screen
        # We take into account the entire height of the map in isometric projection
        # and adjust it so the top is at the center_y, moving it up by half the height of one tile
        total_height_iso = (grid_size_rows * tile_height // 2) * 2  # Total height in isometric view
        offset_y = tile_height*2+tile_height*3+160

        win.fill((0, 0, 0))  # Clear the screen

        def drawChar(row, col):

            tile_type = "CR"
            if (self.last_action == 0):
                tile_type = "CL"
            elif (self.last_action == 1):
                tile_type = "CB"
            elif (self.last_action == 2):
                tile_type = "CR"
            elif (self.last_action == 3):
                tile_type = "CT"
            this_tile_offset_y = tiles[tile_type].get_size()[1] + 16 * scale_factor
            this_tile_offset_x = tiles[tile_type].get_size()[0] / 2
            # Convert grid coordinates to isometric, including walls
            iso_x = (col - row) * (tile_width // 2) + offset_x + tile_width / 2 - this_tile_offset_x
            iso_y = (col + row) * (tile_height // 2) + offset_y - this_tile_offset_y
            win.blit(tiles[tile_type], (iso_x, iso_y))

        current_row = self.state // grid_size_cols
        current_col = self.state % grid_size_cols

        # Render tiles in isometric view, including boundary for walls
        for i in range(-1, grid_size_rows):
            for j in range(-1, grid_size_cols):
                # Determine the type of tile
                if i == -1 or j == -1:
                    if (j) == (-1):  # this must me in minus
                        tile_type = 'WTL'  # Top left wall for the first cell
                    elif (i) == (-1):  # this must be in minus
                        tile_type = 'WTR'  # Top right wall for the last cell in the first row

                else:
                    tile_type = map_layout[i][j]


                this_tile_offset_y = tiles[tile_type].get_size()[1]
                this_tile_offset_x = tiles[tile_type].get_size()[0] / 2

                # Convert grid coordinates to isometric, including walls
                iso_x = (j - i) * (tile_width // 2) + offset_x + tile_width / 2 - this_tile_offset_x
                iso_y = (j + i) * (tile_height // 2) + offset_y - this_tile_offset_y

                win.blit(tiles[tile_type], (iso_x, iso_y))

                if (current_col == j and current_row == i):
                    drawChar(i, j);

            # Define your colors

        #TODO: draw over tall the stuff the image: tiles/frame.png - make it in the middle - its just the same size as window
        # Load the frame image and scale it to the window size
        frame_image = pygame.image.load('tiles/frame.png')

        # Draw (blit) the frame image over everything else
        win.blit(frame_image, (0, 0))

        # Example usage
        self.draw_text_with_gradient(
            "Level "+str(self.level_index)+"  "+str(current_col)+"x"+str(current_row), (30, 604), "font/solstice-nes.ttf", 18,
                                (193, 223, 254),
                                (29, 99, 214))
        # Example usage
        self.draw_text_with_gradient(
            ""+str(self.level_name), (30, 30), "font/solstice-nes.ttf", 18,
            (231, 255, 165),
            (0, 150, 0))

        # Example usage


        self.draw_text_with_gradient(
            "=Train =Evaluate =Reset level =Skin change", (8, 714), "font/solstice-nes.ttf", 18,
            (231, 255, 165),
            (0, 150, 0))





        pygame.display.flip()

    def draw_text_with_gradient(self, text, position, font_path, font_size, top_color, bottom_color):
        global win, win_size

        # Load the custom font
        font = pygame.font.Font(font_path, font_size)

        # Initial X position
        x_pos = position[0]

        # Split the text into segments based on "=" sign
        segments = text.split("=")
        for i, segment in enumerate(segments):
            if i > 0:  # For segments following "=" signs, render the first character in white
                first_char_surface = font.render(segment[0], True, pygame.Color('white'))
                win.blit(first_char_surface, (x_pos, position[1]))
                x_pos += first_char_surface.get_width()
                segment = segment[1:]  # Remove the first character since it's already rendered

            # Render the remaining segment with gradient
            if segment:  # Check if segment is not empty
                text_surface = font.render(segment, True, pygame.Color('white'))
                gradient_surface = pygame.Surface(text_surface.get_size(), pygame.SRCALPHA)
                for y in range(text_surface.get_height()):
                    # Calculate the color for the current position
                    alpha = y / text_surface.get_height()
                    color = [top_color[j] * (1 - alpha) + bottom_color[j] * alpha for j in range(3)]
                    pygame.draw.line(gradient_surface, color, (0, y), (text_surface.get_width(), y))
                gradient_surface.blit(text_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                win.blit(gradient_surface, (x_pos, position[1]))
                x_pos += gradient_surface.get_width()




    def renderOld(self):
        global win, win_size;

        state = self.state;
        # Determine grid size from the map layout
        map_layout = self.map_layout;
        grid_size = len(self.map_layout)
        cell_size = win_size[0] // grid_size
        clock = pygame.time.Clock()

        win.fill((0, 0, 0))  # Clear the screen

        # Draw grid and highlight tiles based on the dynamic map layout
        for i, row in enumerate(map_layout):
            for j, cell in enumerate(row):
                color = (255, 255, 255)  # Default to white for ice
                if cell == 'H':
                    color = (255, 0, 0)  # Red for holes
                elif cell == 'G':
                    color = (0, 255, 255)  # Cyan for goal

                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                pygame.draw.rect(win, color, rect)  # Fill cell
                pygame.draw.rect(win, (255, 255, 255), rect, 1)  # Cell border

        # Highlight current position
        row = state // grid_size
        col = state % grid_size
        highlight_rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
        pygame.draw.rect(win, (0, 255, 0), highlight_rect)  # Green for player

        pygame.display.flip()  # Update the display
        clock.tick(30)  # Cap the frame rate

    def close(self):
        global pygame, win, win_size;
        pygame.quit()
        # Placeholder for any cleanup tasks.
        # For example, if using Pygame for rendering:
        # pygame.quit()
        pass

    def DisableDisplay(self):
        self.enableRendering = False

    def EnableDisplay(self):
        self.enableRendering = True

    def SetTitle(self, param):
        print(param)
        pygame.display.set_caption("Solstice: " + param)
        pass

    def Won(self):
        print("Congratulations, you've reached the goal!")
        self.LoadLevel(self.level_index+1)
        return self.reset()

    def Lost(self):
        print("Oops, you fell into a hole or died!")
        return self.reset()

    def LoadLevel(self, level_index):
        self.level_index = level_index
        self.map_layout = self.load_map(level_index)

