import cv2
import numpy as np
from matplotlib import pyplot as plt

# Helper Method so I know I can see if issue loading
def load_image(path):
	image = cv2.imread(path)
	if image is None: raise ValueError(f"Could not load image from {path}")
	return image


def normalize(image_path, output_path_with_bbox = "output/card_with_bbox.jpg", output_path_card_warped="output/card_warped.jpg"):
	# Load the image
	image = load_image(image_path)

	# Pre process image
	grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(grey, (5, 5), 0)
	edges = cv2.Canny(blurred, 50, 150)

	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Save preprocessed image
	# cv2.imwrite('output/preprocessed.jpg', edges)

	# Find the largest rectangular contour
	largest_contour = None
	max_area = 0
	for contour in contours:
		area = cv2.contourArea(contour)
		if area > max_area:
			# Approximate contour
			perimeter = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
			if len(approx) == 4: # Should have 4 sides for a card
				largest_contour = approx
				max_area = area

	if largest_contour is None:
		raise ValueError("no card")


	pts = largest_contour.reshape(4, 2)
	rect = np.zeros((4, 2), dtype="float32")
	
	# Order the points
	sum = pts.sum(axis=1)
	rect[0] = pts[np.argmin(sum)]  # Top-left
	rect[2] = pts[np.argmax(sum)]  # Bottom-right
	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]  # Top-right
	rect[3] = pts[np.argmax(diff)]  # Bottom-left

	# Calculate the bounding box for portrait alignment
	(tl, tr, br, bl) = rect
	width_a = np.linalg.norm(br - bl)
	width_b = np.linalg.norm(tr - tl)
	height_a = np.linalg.norm(tr - br)
	height_b = np.linalg.norm(tl - bl)

	max_width = max(int(width_a), int(width_b))
	max_height = max(int(height_a), int(height_b))

	# Define the destination points for the portrait rectangle
	dst = np.array([
		[0, 0],
		[max_width - 1, 0],
		[max_width - 1, max_height - 1],
		[0, max_height - 1]
	], dtype="float32")

	# Warp the image
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (max_width, max_height))

	# Re calibrate coordinates of rect to be "portrait"
	# Do this to make bounding box easier to see and not blend with card
	image_with_bbox = image.copy()
	x_vals = list(map(lambda x: x[0], rect.astype(int)))
	y_vals = list(map(lambda x: x[1], rect.astype(int)))
	portrait = [(min(x_vals), max(y_vals)), (max(x_vals), max(y_vals)), (max(x_vals), min(y_vals)), (min(x_vals), min(y_vals))]
	
	# Draw bounding box
	cv2.line(image_with_bbox, tuple(portrait[0]), tuple(portrait[1]), (0, 255, 0), 3)
	cv2.line(image_with_bbox, tuple(portrait[1]), tuple(portrait[2]), (0, 255, 0), 3)
	cv2.line(image_with_bbox, tuple(portrait[2]), tuple(portrait[3]), (0, 255, 0), 3)
	cv2.line(image_with_bbox, tuple(portrait[3]), tuple(portrait[0]), (0, 255, 0), 3)

	# Save the images
	cv2.imwrite(output_path_with_bbox, image_with_bbox)
	cv2.imwrite(output_path_card_warped, warped)

	print(f"Images saved:\n1. Portrait Bounding Box: {output_path_with_bbox}\n2. Warped Card: {output_path_card_warped}")

	return [image_with_bbox, warped]


# # Testing code
# FILE_NAME = 'images/test_pokemon.JPG'
# normalize(FILE_NAME)