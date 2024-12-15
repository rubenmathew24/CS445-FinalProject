# CS445-FinalProject

This project uses a picture of a Magic The Gathering card and gives the user the value of the card.

## Our Process

1. We start by detecting where the card is in the image and isolating it. We warp the image to put it in "portrait mode". This makes the next steps easier to perform.

2. Next we use OCR to determine the name of the card. This is used as a filter so we can limit the amount of cards we are comparing to in the next step.

3. Pull all cards of the same name from a database of MTG cards.

4. Use a simliarity search to determine which exact print of the card the image most likely contains.

5. Return the value of the most similar card

## How to use

### Use GPU
GPU usage for the easyocr package is turned off unless manually turned on. You can go into app.py and manually flip the variable to USE_GPU to enable it.

### Ways of running

We have two ways of running the program. 


#### Single File
Run the app.py file with the image you want to analyze as an argument. It will print the detected card, and it's currrent value in USD.

For example:

```console
  foo@bar:~$ python app.py test.jpg
  Card:  Servant of the Conduit (KLD) 169
  Value: $0.04
```

#### Batch Testing
Running app.py without an argument will use the TESTS dictionary defined in the file. Use this to test the accuracy of the program.

