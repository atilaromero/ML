Uses attention layers to highlight portions of the input. The attentionScores.py creates a html with a colored hexdump, showing in red the areas with greater attention.

To preserve the input size (512 time steps), the convolution layers were changed to use 'same' padding and strides of size 1. This makes the processing time of each sample longer, even if the number of parameter was not greatly increased. Each input byte now is processed at least 8 times more, as the strides had size 8 before.

The final validation accuracy was about the same, despite the increase in processing time. Conclusion: this approach is much slower, and is only worth if the attention scores are necessary.