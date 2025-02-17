FTLLMex5

There were minor issues in the provided code.
Dependency errors and lots of warnings about using depreceated libraries, but these did not effect today. 

One wrong call because rogue did not have the attribute --version-- so it needed a different way of dispalying it

The training process had wonky device handling in the case of a two GPU training environment that kaggle offers. This was solved by selecting the one GPU env:))

The testing code had some weird way of handling the models and I am not sure of the horrenodus results are because of this. I did do some changes to it, at first you were comparing
the finetuned model with itself and not the original model which is obviously weird? 

The "näyttökuva" files are screenshots which contain the test results
