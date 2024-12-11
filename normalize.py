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
	edges = cv2.Canny(blurred, 30, 100)

	# Clean up the edges (GPT proposed using dilate/erode methods)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	edges = cv2.dilate(edges, kernel, iterations=3)
	edges = cv2.erode(edges, kernel, iterations=2)

	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Save Image of Contours Drawn
	debug = image.copy()
	cv2.imwrite('debug/contours.jpg', cv2.drawContours(debug, contours, -1, (0,255,0), 3))

	# Save preprocessed image
	# cv2.imwrite('debug/preprocessed.jpg', edges)

	# Find the largest rectangular contour
	largest_contour = None
	max_area = 0
	for contour in contours:
		area = cv2.contourArea(contour)
		if area > max_area:
			# Approximate contour
			perimeter = cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
			if len(approx) == 4: # Should have 4 sides for a card
				largest_contour = approx
				max_area = area

	if largest_contour is None:
		raise ValueError("no card")


	# print(largest_contour)

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

	# Bounding box
	image_with_bbox = image.copy()
	rect = rect.astype(int)

	cv2.line(image_with_bbox, tuple(rect[0]), tuple(rect[1]), (0, 255, 0), 3)
	cv2.line(image_with_bbox, tuple(rect[1]), tuple(rect[2]), (0, 255, 0), 3)
	cv2.line(image_with_bbox, tuple(rect[2]), tuple(rect[3]), (0, 255, 0), 3)
	cv2.line(image_with_bbox, tuple(rect[3]), tuple(rect[0]), (0, 255, 0), 3)

	# Save the images
	cv2.imwrite(output_path_with_bbox, image_with_bbox)
	cv2.imwrite(output_path_card_warped, warped)
	# print("Images Saved")

	return [image_with_bbox, warped]


# # Testing code
# FILE_NAME = ''
# normalize(FILE_NAME)