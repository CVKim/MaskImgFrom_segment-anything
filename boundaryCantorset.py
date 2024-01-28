import cv2
import numpy as np
import random
import time
import os

random.seed(int(time.time()))

def find_defect_mask_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, w, h

def find_partial_valid_positions(boundary, mask_bounding_box, num_positions):
    valid_positions = []
    boundary_coords = np.where(boundary == 255)
    min_x, max_x = np.min(boundary_coords[1]), np.max(boundary_coords[1])
    min_y, max_y = np.min(boundary_coords[0]), np.max(boundary_coords[0])
    mask_x, mask_y, mask_width, mask_height = mask_bounding_box

    for _ in range(num_positions):
        while True:
            x = random.randint(min_x, max_x - mask_width)
            y = random.randint(min_y, max_y - mask_height)

            overlap_region = boundary[y:y + mask_height, x:x + mask_width]
            if np.any(overlap_region == 255):  # Check for any overlap
                valid_positions.append((x, y))
                break

    return valid_positions

hazelnut_boundary_mask = cv2.imread("000_boudary.bmp", cv2.IMREAD_GRAYSCALE)
defect_mask = cv2.imread("000_mask.png", cv2.IMREAD_GRAYSCALE)

mask_bounding_box = find_defect_mask_bounding_box(defect_mask)
valid_positions = find_partial_valid_positions(hazelnut_boundary_mask, mask_bounding_box, 100)

out_dir = 'data/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

output_image_paths = []
for image_index, (x, y) in enumerate(valid_positions):
    result_image = np.zeros_like(hazelnut_boundary_mask)
    mask_x, mask_y, mask_width, mask_height = mask_bounding_box
    result_image[y:y + mask_height, x:x + mask_width] = defect_mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width]
    
    if np.any(result_image == 255):  # Check if there is any defect mask region in the result image
        output_path = f"{out_dir}output_image_{image_index:03d}.png"
        cv2.imwrite(output_path, result_image)
        output_image_paths.append(output_path)

print(output_image_paths[:5])
