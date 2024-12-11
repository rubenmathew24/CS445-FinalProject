import normalize
import ocr

IMAGE = "images/Green.JPG"

# Off by Default 
USE_GPU = False

def main():

	# Load Image
	original_image = normalize.load_image(IMAGE)

	# Normalize Image
	bbox, warped = normalize.normalize(IMAGE)

	# Apply OCR
	name = ocr.get_card_name(warped, USE_GPU)

	# Look Up Name

	# Apply Similarity Search

	# Return value of Card

	return name # Placeholder until rest of method is implemented

if __name__ == "__main__":
	main()
