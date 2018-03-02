# Latent Similarity Networks
The LSN is a neural network that learns to encode images to latent space vectors, then uses the cosine similarity of an image's latent representation with previous training data to match it to a class. Ultimately, it is able to classify images given a small amount of training data.

Although definitely not state-of-the-art, I managed to achieve ~85% accuracy on the Omniglot dataset ( the mnist of one-shot learning ) as well as ~89% accuracy on Adience DB's dataset of face images. Both of these were done with 5 classes per episode.
