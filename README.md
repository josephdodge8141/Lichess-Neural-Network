## Lichess Neural Network

#### Description

This is a neural network that is going to be trained on games from the lichess database. It will be given every position from every game based on the lichess PGN and output the expected score of the postion (between 0 and 1). It will then take a weighted sum of its guesses and compare it to the actual result of the game (1, 0.5 or 0). This will be used to determine the error. Because it is using the result and not another evaluation nor including the time in our evaluation, the data will not include games that are bullet or were won by flagging. To avoid bad data it will also not look at games played by players below 1600.

#### File Structure

* Lichess Neural Network
    * utils
        * other
        * converters
    * tests
        * utils
        * src
    * src
        * code for NN
    * Docs
        * arch 
        * patterns

#### Understand the pages

The utils will contain all reusable functions to avoid rewriting any code. There will be a README inside listing all of the functions and how to use them. The converters folder is specifically meant for functions that change something into something else.

Tests are divided into utils and src. For readability these are separated. Tests using tensorflow for the NN are written differently than the ones written with the python pattern for the utils.

The docs are separated into architecture and patterns. Arch will contain all documents relating to the general architecture of the NN including, pictures of teh design and reasoning for major decisions. Patterns will contain naming patterns for the site including folders, pages, classes, methods, and variables. It will also include patterns for writing tests and when to put something in the utils folder.

