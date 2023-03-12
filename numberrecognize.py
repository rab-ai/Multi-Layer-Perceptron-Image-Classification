import math

def relu(x):
    l = []
    for i in x: l.append(0) if i<0 else l.append(i)
    return l

def sigmoid(x):
    l = []
    for i in x: l.append(0) if i<=-700 else l.append(1/(1+math.exp(-i))) if (i>-700 and i<700) else l.append(1)
    return l

def multiply(a,b):
    return a*b

def linear(x,w):
    linear_list = []
    for i in w:
        a = list(map(multiply, x, i))
        linear_list.append(sum(a))
    return linear_list

def forward_pass(network, x_sample):
    x = x_sample
    for i in network:
        if type(i) == str:
            if i.find("relu") != -1:
                x = (relu(x)) 
            elif i.find("sigmoid") != -1:
                x = (sigmoid(x))  
        if type(i) == list:
            for j in i:
                if type(j) == str:
                    if j.find("linear") != -1:
                        x =(linear(x, i[1]))                  
    return x
