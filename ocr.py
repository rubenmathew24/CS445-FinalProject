import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

def get_card_name(normalized_img, use_gpu = False):

	# Crop image
	cropped_image = normalized_img[:int(0.2 * normalized_img.shape[0]), :]

	# Display the cropped image
	# plt.figure(figsize=(12, 6))
	# plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
	# plt.axis('off')
	# plt.title("Cropped")
	# plt.show()

	# OCR implementation
	reader = easyocr.Reader(['en'], gpu = use_gpu, verbose=False)
	results = reader.readtext(cropped_image)

	# No Results
	if len(results) < 1: 
		raise ValueError("no text found")
	

	best_fit = 0
	# Print All Results if too many, return the largest one
	if len(results) > 1:

		for i, tup in enumerate(results):
			bbox, text, confidence = tup
			# print(f'{i}.')
			# print("\t * " + str(bbox))
			# print("\t * " + str(text))
			# print("\t * " + str(confidence))

			if len(text) > len(results[best_fit]):
				best_fit = i
		# raise ValueError("too many results found")
	
	# Found 1 result (expected)
	bbox, text, confidence = results[best_fit]

	# print(str(bbox))
	# print(str(text))
	# print(str(confidence))

	# Draw bounding box test
	text_bbox = normalized_img.copy()

	cv2.line(text_bbox, tuple(bbox[0]), tuple(bbox[1]), (0, 255, 0), 3)
	cv2.line(text_bbox, tuple(bbox[1]), tuple(bbox[2]), (0, 255, 0), 3)
	cv2.line(text_bbox, tuple(bbox[2]), tuple(bbox[3]), (0, 255, 0), 3)
	cv2.line(text_bbox, tuple(bbox[3]), tuple(bbox[0]), (0, 255, 0), 3)

	cv2.imwrite("debug/text_with_bbox.jpg", text_bbox)

	return text

# Test code
# warped_path = 'output/card_warped.jpg'
# warped = normalize.load_image(warped_path)
# name = get_card_name(warped)
