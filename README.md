# Multi-Layer-Perceptron-Image-Classification
Using Forward-Pass in a Deep Network

In this project, I make inference using a pre-trained deep
network, namely a Multi-layer Perceptron (MLP). To be specific, we look at the problem of
image classification where we assign a label y (e.g. ‘car’, ‘house’, ‘smiling’, ‘digit 0’) to an image
x provided as input. For example, the image can be the image of a digit (as shown in the picture below, namely Figure 1),
and the estimated label can be digit 4. To provide our input to an MLP, we first transform
the image into a 1D vector, as illustrated in Figure 1. For this, the image is scanned row by row
and the concatenation of all the rows is taken as the vector representation of the image. Then,
the network will provide us prediction score for each class. Finally, our estimated label, y-pred,
will be the class with the highest score.
![2023-03-12_05-45-45](https://user-images.githubusercontent.com/89254644/224521348-46e761a1-bd67-44b6-b763-d89ea5671238.png)

The deep network shema:

![2023-03-12_05-47-37](https://user-images.githubusercontent.com/89254644/224521398-3f60c0d1-a36f-4aca-9606-b25913747f4c.png)

– mnist.pkl: A file containing 1000 digit images and their labels.\
– network_3layer.pkl: A file for a pre-trained MLP with 3 linear layers.\
– network_2layer.pkl: A file for a pre-trained MLP with 2 linear layers.\
– utils.py: A Python module with a set of utilities load_dataset, load_network, display_image, display_network, calculate_accuracy.

To get the result, run main.py. The order of the output will be in the following:
1. Visualising the image using ASCII chars.
2. The correct label for the image above.
3-4. An array displaying a total of 10 float numbers. The index of the highest number is the correct label for the image.
5-6. The accuracy percent of networks by using calculate_accuracy function in utility for network1 and network2.

Example 1:
![2023-03-12_06-04-25](https://user-images.githubusercontent.com/89254644/224521920-f5b22d51-7f3e-4a10-a3fe-12cbb78f47ea.png)

Example 2:
![2023-03-12_06-05-59](https://user-images.githubusercontent.com/89254644/224521958-2fe070b1-be06-47bc-956b-635aa109713e.png)

In example 1, 6. index of both of the arrays is the highest value, this shows that the number displaying above is six. Additionally, in example 2 index with one of both of the arrays is the highest value, so it shows that the number displaying above is 1.

For the other images, we can change the second index of dataset["X_test"][] and dataset["y_test"][] from 0to 784 and examine the resulting arrays.
