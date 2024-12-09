# CS445-FinalProject
 
This project uses a picture of a Magic The Gathering card and gives the user the value of teh card.

## Our Process:

1. We start by detecting where the card is in the image and isolating it. We warp the image to put it in "portrait mode". This makes the next steps easier to perform.

2. Next we use OCR to determine the name of the card. This is used as a filter so we can limit the amount of cards we are comparing to in the next step.

3. Pull all cards of the same name from a database of MTG cards.

4. Use a simliarity search to determine which exact print of the card the image most likely contains.

5. Return the value of the most similar card


## How to use:

1. Go into app.py

2. Change the IMAGE variable to the path of the image you want to check

3. Run app.py
