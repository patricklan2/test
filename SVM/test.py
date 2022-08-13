import numpy as np
PATH = "../MNIST.npz"
mnist = np.load(PATH)
images,labels = mnist["train_images"],mnist['train_label']
images = images.reshape(-1,28*28)/255
labels = (labels==0)*2-1

smo = np.load('./SMOmnist.npz')
alpha_y,b,train_x = smo['alpha_y'],smo['b'],smo['x']

def K_vector(x):
    return np.array([np.exp(-np.linalg.norm(x-i,ord=2)/200) for i in train_x])

def g(x):
    return np.dot(alpha_y,K_vector(x)) + b

result = np.array([g(x) for x in images[:1000]])
result = (result>0)*2-1
acc = result==labels[:1000]
print(acc.mean())
