# kwikpic
discription on how the emotion.py was trained
the model is train using CNN model.Strat with the initialization of the model 
followed by batch normalization layer and then deffernt convents layers with ReLu as an
activation function,max pool layers and dropouts to do learning efficiently.
we compile the model using Adam an an optimizer loss as categories cross-entropy amd metrics as accuarcy 
after compilinig the model we fit then fit the data for training and validation. we 
are taking the batch size to be 32 with 30 epochs.
