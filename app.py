import normalize
import ocr
import matching
from tqdm import tqdm

IMAGE = "images/Red.jpg"

# Off by Default
USE_GPU = False

# Batch Testing
PERFORM_TESTING = False

TESTS = {
    "Cloud Elemental (M11) 50": 'Blue.JPG',
    "Flesh to Dust (M15) 98": 'Black.JPG',
    "Servant of the Conduit (KLD) 169": 'Green.JPG',
    "Battering Craghorn (ONS) 188": 'Red.JPG',
    "Blightbelly Rat (ONE) 289": 'Special.JPG',
    "Fleeting Spirit (VOW) 14": 'White.JPG',
    "Cloudblazer (KLD) 176": 'IMG_2348.jpg',
    "Armored Wolf-Rider (DGM) 52": 'IMG_2349.jpg',
    "Morgue Burst (DGM) 86": 'IMG_2350.jpg',
    "Deputy of Acquittals (DGM) 65": 'IMG_2351.jpg',
    "Viashino Firstblade (DGM) 113": 'IMG_2352.jpg',
    "Beetleform Mage (DGM) 54": 'IMG_2353.jpg',
    "Transluminant (RAV) 186": 'IMG_2354.jpg',
    "Arcbound Shikari (MH2) 184": 'IMG_2355.jpg',
}


def main(testing=False):

    # Load Image
    original_image = normalize.load_image(IMAGE)

    # Normalize Image
    bbox, warped = normalize.normalize(IMAGE)

    # Apply OCR
    name = ocr.get_card_name(warped, USE_GPU)

    # Look Up Name
    name = matching.find_closest_match(name)

    # Apply Similarity Search
    printing, scryfall_id = matching.match_printing(warped, name)

    # Return value of Card
    value = matching.get_card_value(scryfall_id)

    return printing, value


def batch_testing():

    # Testing names right now, until rest is implemented
    passed = 0
    failed = []

    for name, file in tqdm(TESTS.items(), desc="Testing", leave=False):
        global IMAGE
        IMAGE = 'images/' + file
        computed, value = main()

        try:
            assert name == computed
            passed += 1
        except:
            failed += [f'Known Name: {name} doesn\'t match Computed Name: {computed}']

    # Passed Cases
    percent = round(100*passed/len(TESTS.items()), 2)
    print(f'Passed {percent}% of cases')

    # Failed Cases
    if len(failed) > 1:
        print("Failed Cases:")
    for idx, item in enumerate(failed):
        print(f'\t{idx}. {item}')


if __name__ == "__main__":
    if PERFORM_TESTING: batch_testing()
    else:
        name, val = main()
        print(name + " which is worth $" + val)