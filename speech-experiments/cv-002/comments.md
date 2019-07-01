Experiment cv-002 uses a source code based on experiment v-002, just changing the dataset and batch size.

A key difference between experiment cv-001 and cv-002 is that in the first, the ``fit'' function was called many times, one for each epoch, while in the later, the ``fit'' function was called only once, specifying the maximum number of epochs that should be run. The disadvantage of calling the ``fit'' function many times is that the internal state of the optimizer algorithm is reset at each interaction, thus causing the observed stagnation of the loss value.

Using a model previously trained with vowels, it achieved 0.90 accuracy on 140 syllables in 4301 epochs , taking 8m6s.
