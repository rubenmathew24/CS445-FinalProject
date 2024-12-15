import requests
import numpy as np
import cv2
import homography
import matplotlib.pyplot as plt


def find_closest_match(name):
    return requests.get(f'https://api.scryfall.com/cards/named?fuzzy={name}').json()['name']


def score_aligned(im1, im2):
    score = 0
    score += cv2.matchTemplate(im1, im2, cv2.TM_CCOEFF_NORMED)[0][0]

    # bonus for matching set symbol
    symbol_location = (int(im2.shape[0] * .86), int(im2.shape[1] * .57))
    symbol_size = (int(im2.shape[0] * .06), int(im2.shape[1] * .04))

    im1_symbol = im1[symbol_location[0]:symbol_location[0] +
                     symbol_size[0], symbol_location[1]:symbol_location[1] + symbol_size[1]]
    im2_symbol = im2[symbol_location[0]:symbol_location[0] +
                     symbol_size[0], symbol_location[1]:symbol_location[1] + symbol_size[1]]

    score += cv2.matchTemplate(im1_symbol, im2_symbol,
                               cv2.TM_CCOEFF_NORMED)[0][0] * 0.2

    return score


def score_pair(im1, im2):
    # align images
    im1 = cv2.resize(im1, (im2.shape[1] - 50, im2.shape[0] - 50))
    H = homography.auto_homography(im1, im2, homography.computeHomography)
    im1_aligned = cv2.warpPerspective(im1, H, (im2.shape[1], im2.shape[0]))

    # show images
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(im2)
    # plt.subplot(1, 3, 2)
    # plt.imshow(im1)
    # plt.subplot(1, 3, 3)
    # plt.imshow(im1_aligned)
    # plt.show()

    # calculate score
    return score_aligned(im1_aligned, im2)


def match_printing(warped, name):
    printingssearch = requests.get(
        f'https://api.scryfall.com/cards/named?fuzzy={name}').json()['prints_search_uri']

    printings = requests.get(printingssearch).json()['data']

    printings = [
        printing for printing in printings if 'paper' in printing['games']]

    max_score = 0
    best_match = None

    for printing in printings:
        image_url = printing['image_uris']['normal']
        image = cv2.imdecode(np.frombuffer(requests.get(
            image_url).content, np.uint8), cv2.IMREAD_COLOR)
        score = score_pair(warped, image)

        if score > max_score:
            max_score = score
            best_match = printing

    return f'{best_match["name"]} ({best_match["set"].upper()}) {best_match["collector_number"]}'
