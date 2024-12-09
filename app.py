import normalize

IMAGE = "images/test_pokemon.JPG"


def main():
    
	# Load Image
    original_image = normalize.load_image(IMAGE)
    
	# Normalize Image
    bbox, warped = normalize.normalize(original_image)
    
	# Apply OCR
    
	# Look Up Name
    
	# Apply Similarity Search
    
	# Return value of Card
    

if __name__ == "__main__":
    main()
