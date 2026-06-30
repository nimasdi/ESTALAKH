import cv2
import numpy as np

from src.grid_extraction import CELL_SIZE, WARP_SIZE, GridExtraction


def _draw_centered_digit(canvas, row, col, text, color, font, thickness):
    cell_center_x = (col + 0.5) * CELL_SIZE
    cell_center_y = (row + 0.5) * CELL_SIZE
    
    # ~60% of the cell height
    scale = cv2.getFontScaleFromHeight(font, int(CELL_SIZE * 0.6), thickness)
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    org = (int(cell_center_x - tw / 2), int(cell_center_y + th / 2))
    cv2.putText(canvas, text, org, font, scale, color, thickness, cv2.LINE_AA)


def render_solution( image, extraction: GridExtraction, solved_grid, given_mask, color=(0, 180, 0), thickness= 3):
    
    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    canvas = np.zeros((WARP_SIZE, WARP_SIZE, 3), np.uint8)
    for row in range(9):
        for col in range(9):
            if given_mask[row][col]:
                continue
            value = solved_grid[row][col]
            if value == 0:
                continue
            _draw_centered_digit(canvas, row, col, str(value), color, font, thickness)

    warped_back = cv2.warpPerspective(canvas, extraction.inverse_matrix, (width, height))

    overlay_mask = warped_back.any(axis=2)
    out = image.copy()
    out[overlay_mask] = warped_back[overlay_mask]
    return out
