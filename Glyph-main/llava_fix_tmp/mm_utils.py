import ast


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    if isinstance(grid_pinpoints, str):
        grid_pinpoints = ast.literal_eval(grid_pinpoints)

    width, height = image_size
    best_width, best_height = min(
        grid_pinpoints,
        key=lambda point: abs(point[0] - width) + abs(point[1] - height),
    )
    return best_width // patch_size, best_height // patch_size
