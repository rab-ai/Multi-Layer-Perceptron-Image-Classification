import utils
dataset = utils.load_dataset("mnist.pkl")
network1 = utils.load_network("network_3layer.pkl")
network2 = utils.load_network("network_2layer.pkl")

#utils.display_network(network1)
#utils.display_network(network2)

X_sample = dataset["X_test"][50]
utils.display_image(X_sample)
print(dataset["y_test"][50])


from numberrecognize import *
forward_pass(network1, X_sample)
forward_pass(network2, X_sample)
print(utils.calculate_accuracy(dataset, network1, predictor=forward_pass))
print(utils.calculate_accuracy(dataset, network2, predictor=forward_pass))