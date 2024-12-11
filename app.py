import normalize
import ocr
from tqdm import tqdm

IMAGE = "images/Green.JPG"

# Off by Default 
USE_GPU = False

TESTS = {
    "Fleeting Spirit" : 'White.JPG',
    "Battering Craghorn" : 'Red.JPG',
    "Servant of the Conduit" : 'Green.JPG',
    "Flesh to Dust" : 'Black.JPG',
    "Blightbelly Rat" : 'Special.JPG',
	"Cloud Elemental" : 'Blue.JPG',
}

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

def batch_testing():
    
    # Testing names right now, until rest is implemented
    passed = 0
    failed = []

    for name, file in tqdm(TESTS.items(), desc="Testing", leave=False):
        global IMAGE
        IMAGE = 'images/' + file
        computed = main()

        try: 
            assert name == computed
            passed += 1
        except:
            failed += [f'Known Name: {name} doesn\'t match Computed Name: {computed}']

    # Passed Cases
    percent = round(100*passed/len(TESTS.items()), 2)
    print(f'Passed {percent}% of cases')

    # Failed Cases
    if len(failed) > 1: print("Failed Cases:")
    for idx, item in enumerate(failed):
        print(f'\t{idx}. {item}')

if __name__ == "__main__":
	# main()
    batch_testing()
