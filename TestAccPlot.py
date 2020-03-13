import numpy as np 
import matplotlib.pyplot as plt

AccData = np.load('TestAccs.npy')

epochs = len(AccData)


plt.plot(list(range(1,epochs+1)), AccData)
plt.title('Test Set Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.show()

