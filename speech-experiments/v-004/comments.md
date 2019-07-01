Experiment v-004 adds a maxpool layer after each convolutional layer. The ``data\_format=`channels\_first`'' option specifies the dimension that should be reduced, which should be the data of a single time step. With the default option the MaxPool1D function would reduce the number of time steps.

It achieved 1.00 accuracy in 501 epochs, taking 0m40s.