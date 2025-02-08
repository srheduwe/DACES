import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

def grid_downsizer(image: torch.Tensor, dim: int, scaling_factor: int, popsize: int = 100):
    '''Returns downscaled image, number of pixels and grid.
    The grid can be used to upscale the problem to the original size.
    If scaling is 1, it returns the original image and dimensions and no grid.
    If, e.g. scaling=3, the downscaled image consists of the center pixel of a 3x3 grid.'''
    
    if scaling_factor == 1:
        return image, dim, dim, None
    else:
        pixel_number = round(dim / scaling_factor)
        grid_y, grid_x = torch.meshgrid(torch.arange(pixel_number), torch.arange(pixel_number), indexing='ij')  # Create a grid of indices for height and width
        grid_y = (grid_y * scaling_factor + scaling_factor // 2).int()  # Scale the grid indices to match the original image size
        grid_x = (grid_x * scaling_factor + scaling_factor // 2).int()
        grid_c = torch.arange(3).reshape(3, 1, 1)  # Create a grid for the channels
        downscaled_image = image[:, grid_y, grid_x]  # Use the grid to downscale the image
        grid_b = torch.arange(popsize).reshape(popsize, 1, 1, 1)  # Add extra grid of popsize for upscaling

        # Remove the extra dimension added to grid_y and grid_x for broadcasting
        grid_y = grid_y.squeeze(0)
        grid_x = grid_x.squeeze(0)

        return downscaled_image, pixel_number, pixel_number, [grid_b, grid_c, grid_y, grid_x]


def gaussian_blur(image, kernel_size=5, sigma=1):
    """Applies Gaussian blur to an image."""
    channels, height, width = image.shape
    kernel = torch.zeros((kernel_size, kernel_size), device="cuda")
    for x in range(-kernel_size//2 + 1, kernel_size//2 + 1):
        for y in range(-kernel_size//2 + 1, kernel_size//2 + 1):
            kernel[x + kernel_size//2, y + kernel_size//2] = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * sigma**2)))
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    image = image.unsqueeze(0)
    blurred_image = F.conv2d(image, kernel, padding=kernel_size//2, groups=channels)
    return blurred_image.squeeze(0)

def subspace_activation(image: torch.Tensor, popsize: int = 100):
    '''Returns downscaled image, number of pixels and grid.
    The grid can be used to upscale the problem to the original size.
    This function finds the largest binary island in the red channel of an image.
    The downscaled height and width are at least 50% smaller, but always greater than 8x8.'''

    # Smooth the image by averaging the pixel values
    smoothed_image = gaussian_blur(image, kernel_size=5, sigma=1)

    # Use red channel only for identifying subspace
    binarized_tensor = (smoothed_image[0] > 0.5).float()
    H, W = binarized_tensor.shape
    min_height, min_width = 8, 8
    visited = torch.zeros_like(binarized_tensor, dtype=torch.bool)
    max_size = 0
    best_rect = None

    def fill(i, j):
        queue = [(i, j)]
        island = []
        while queue:
            x, y = queue.pop(0)
            # Check if current coordinates are out of bounds, already visited, or not part of the island
            if x < 0 or x >= H or y < 0 or y >= W or visited[x, y] or binarized_tensor[x, y] == 0:
                continue
            visited[x, y] = True
            island.append((x, y))
            # Add neighboring coordinates
            queue.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])
        return island

    for i in range(H):
        for j in range(W):
            if binarized_tensor[i, j] == 1 and not visited[i, j]:
                island = fill(i, j)
                if len(island) > max_size:
                    max_size = len(island)
                    min_y = min(island, key=lambda x: x[0])[0]
                    max_y = max(island, key=lambda x: x[0])[0]
                    min_x = min(island, key=lambda x: x[1])[1]
                    max_x = max(island, key=lambda x: x[1])[1]
                    best_rect = (min_x, max_x, min_y, max_y)

    # If there is no island, we just take the middle of the picture
    if best_rect is None:
        min_x = int(round((W / 2) - 4))
        max_x = int(round((W / 2) + 4))
        min_y = int(round((H / 2) - 4))
        max_y = int(round((H / 2) + 4))
        best_rect = (min_x, max_x, min_y, max_y)

    min_x, max_x, min_y, max_y = best_rect

    # Ensure minimum/maximum height and width
    width = max_x - min_x
    height = max_y - min_y
    height = max(height, min_height)
    width = max(width, min_width)

    # Correction when window is out of picture range
    correction_h = max(0, min_y + height - H)
    correction_w = max(0, min_x + width - W)
    min_y -= correction_h
    min_x -= correction_w

    col_indices = torch.arange(min_y, min_y + height)
    row_indices = torch.arange(min_x, min_x + width)
    channel_indices = torch.tensor([0, 1, 2])
    grid_b = torch.arange(popsize)  # Add extra grid of popsize for upscaling

    # Create the grid suitable for broadcasting
    grid = torch.meshgrid(grid_b, channel_indices, col_indices, row_indices, indexing='ij')

    return image[:, min_y:min_y + height, min_x:min_x + width], len(col_indices), len(row_indices), list(grid)


def nearest_neighbor_interpolation(image, scaling_factor):
    scaling_factor = 1/scaling_factor
    C, H, W = image.shape                       # Get the dimensions of the original image
    new_H = int(H * scaling_factor)             # Calculate the dimensions of the scaled image
    new_W = int(W * scaling_factor)
    
    grid_y, grid_x = torch.meshgrid(torch.arange(new_H), torch.arange(new_W))     # Create a grid for the new coordinates   
    grid_y = (grid_y // scaling_factor).to(torch.long)                                
    grid_x = (grid_x // scaling_factor).to(torch.long)
    
    grid_y = torch.clamp(grid_y, 0, H - 1)      # Clip the coordinates to be within the original image dimensions
    grid_x = torch.clamp(grid_x, 0, W - 1)
    scaled_image = image[:, grid_y, grid_x]     # Use the computed grid to index the original image
    
    def inverse_scaling(scaled_images):
        original_H = H
        original_W = W
        original_grid_y, original_grid_x = torch.meshgrid(torch.arange(original_H), torch.arange(original_W))
        scaled_grid_y = (original_grid_y * scaling_factor).to(torch.long)
        scaled_grid_x = (original_grid_x * scaling_factor).to(torch.long)
        scaled_grid_y = torch.clamp(scaled_grid_y, 0, new_H - 1)
        scaled_grid_x = torch.clamp(scaled_grid_x, 0, new_W - 1)
        restored_images = scaled_images[:, :, scaled_grid_y, scaled_grid_x]
        return restored_images
    
    return scaled_image, new_H, new_W, inverse_scaling

def bilinear_interpolation(image: torch.tensor, dim: int, scaling_factor: int = 32):
    dim_down = np.ceil(dim/scaling_factor).astype(int)
    downscaling = transforms.Compose([transforms.Resize([dim_down, dim_down])])
    downscaled_image = downscaling(image)
    grid = transforms.Compose([transforms.Resize([dim, dim])])
    return downscaled_image, dim_down, dim_down, grid

def downsizer(config, image: torch.tensor, dim: int, popsize: int = 100):
    '''Wrapper for downsizing functions.'''
    if config.experiment.scaling_method == "grid":
        downscaled_image, height, width, grid = grid_downsizer( # Scaling down the problem, if necessary
                                                image=image,
                                                dim=dim,
                                                scaling_factor=config.parameters.scaling_factor,
                                                popsize=popsize)
    elif config.experiment.scaling_method == "subspace_activation":
        downscaled_image, height, width, grid = subspace_activation(image=image, popsize=popsize)
    elif config.experiment.scaling_method == "bilinear_interpolation":
        downscaled_image, height, width, grid = bilinear_interpolation(image=image, dim=dim, scaling_factor=config.parameters.scaling_factor)
    elif config.experiment.scaling_method == "nni":
        downscaled_image, height, width, grid = nearest_neighbor_interpolation(image=image, scaling_factor=config.parameters.scaling_factor)
    return downscaled_image, height, width, grid