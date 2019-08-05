This is a script that'll accept a CSV file and create a cross validated classification
model for it.

Written in Python 2.7 for no particular reason. It may work in 3.* anyways, haven't tested that.

First, I created a simple setup that loads every feature as a categorical variable,
liberally discards rows it doesn't understand and one-hots everything. I was drawn towards
creating this as a simple diagnostic script to run on any mixed dataset, so I didn't want to
hardcode any feature masks/selectors. I added sparse scaling and naive dimensionality reduction
to the pipeline. Sparse scaling would normalize the database ID that remained as a feature.
I wanted to help it get blown out during the DR step. DR would also hopefully cull any other
abhorrently superfluous one-hot features. After a few different runs I settled on
using a multi-layer perceptron.