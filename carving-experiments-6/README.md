Tries to differentiate ini, mid, and end sectors, by creating categories like pdf-ini, pdf-mid, and pdf-end.

Not very successful strategy. Maybe it is better to train a perceptron with first sectors. To train to recognize end sectors, an idea is to train a network to guess the correct mask for a sector.