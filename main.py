import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from torchvision import datasets, transforms
import torchvision
import pdb
from sklearn.metrics import roc_curve, auc

epochs = 100
K = 5

#loading the data in, batch size = minibatch size
#important to have the data in its own folder with labels in diff folders as is done here
loader = torch.utils.data.DataLoader(
datasets.ImageFolder("data",
transform = transforms.Compose([transforms.RandomResizedCrop(256, scale=(0.95,1.0), ratio = (0.98,1.02)), transforms.RandomRotation(5,resample=2), transforms.RandomPerspective(distortion_scale=0.05), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]) #reshape to 512x512 b/c they were unequal and they're all around 512
),
batch_size = 10, shuffle = True
)

train_loader = torch.utils.data.DataLoader(
datasets.ImageFolder("train",
transform = transforms.Compose([transforms.RandomResizedCrop(256, scale=(0.95,1.0), ratio = (0.98,1.02)), transforms.RandomRotation(5,resample=2), transforms.RandomPerspective(distortion_scale=0.05), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]) #reshape to 512x512 b/c they were unequal and they're all around 512
),
batch_size = 20, shuffle = True
)

val_loader = torch.utils.data.DataLoader(
datasets.ImageFolder("val",
transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]) #reshape to 512x512 b/c they were unequal and they're all around 512
),
batch_size = 10, shuffle = False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ResBlock(nn.Module):
	def __init__(self, in_, mid_, out_, k, DO = 0.0, ds = 0):
		super().__init__()

		self.DO = DO
		self.pad = ((k-1)//2, k//2, (k-1)//2, k//2)

		'''
		self.conv = nn.Sequential(
		nn.ReflectionPad2d(self.pad),
		nn.Conv2d(in_, out_, k),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(out_), nn.Dropout2d(self.DO),
		nn.ReflectionPad2d(self.pad),
		nn.Conv2d(out_, out_, k),
		nn.ReLU(inplace=True),
		nn.BatchNorm2d(out_), nn.Dropout2d(self.DO)
		)
		'''

		self.conv = nn.Sequential(
		nn.ReflectionPad2d(self.pad),
		nn.Conv2d(in_, mid_, 1),
		nn.LeakyReLU(inplace=True),
		nn.BatchNorm2d(mid_),

		nn.Conv2d(mid_, mid_, k),
		nn.LeakyReLU(inplace=True),
		nn.BatchNorm2d(mid_),

		nn.Conv2d(mid_, out_, 1),
		nn.LeakyReLU(inplace=True),
		nn.BatchNorm2d(out_),
		)

		self.ds = ds
		if self.ds > 0:
			self.DS = nn.Sequential(
			nn.Conv2d(out_,out_,self.ds, self.ds), nn.LeakyReLU(inplace=True),
			nn.BatchNorm2d(out_))

	def forward(self, input):
		out = self.conv(input)
		out += input

		if self.ds > 0:
			out = self.DS(out)

		return out



class Net(nn.Module):
	def __init__(self, size):
		super(Net, self).__init__()

		self.in_ = 128 #number of channels 
		self.mid_ = 64
		self.out_ = self.in_
		self.featureExtractor = nn.Sequential(
			nn.Conv2d(size, 32,1),
			ResBlock(32, 16, 32, 4, ds = 2), #4 refers to kernel size, convolution size of 4x4 
			nn.Conv2d(32, 64, 1),
			ResBlock(64, 32, 64, 4),
			ResBlock(64, 32, 64, 4),			
			ResBlock(64, 32, 64, 4, ds = 2),
			ResBlock(64, 32, 64, 4),
			ResBlock(64, 32, 64, 4),
			ResBlock(64, 32, 64, 4),
			nn.Conv2d(64,128,1),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4, ds = 2),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4, ds = 2),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4, ds = 2),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4),
			ResBlock(self.in_, self.mid_, self.out_, 4, ds = 2))

		self.dense = nn.Sequential(
			nn.Linear(2048, 256),
			nn.ReLU(inplace = True),
			nn.BatchNorm1d(256),			
			nn.Linear(256, 128),
			nn.ReLU(inplace = True),
			nn.BatchNorm1d(128),
			nn.Linear(128, 64),
			nn.ReLU(inplace = True),
			nn.BatchNorm1d(64),
			nn.Linear(64, 32),
			nn.ReLU(inplace = True),
			nn.BatchNorm1d(32),		
			nn.Linear(32, 3))



		self.AA = nn.AdaptiveAvgPool2d((4,4))

	def forward(self, x):
		x = self.featureExtractor(x)
		#pdb.set_trace()
		#x = self.AA(x)
		x = torch.flatten(x, 1)
		x = self.dense(x)

		return x

#have to
#model = Net(3).to(device)

#model = torch.hub.load('pytorch/vision:v0.5.0', 'resnext101_32x8d', pretrained=True).to(device)
model = torchvision.models.resnext101_32x8d(pretrained=False).to(device)
model.avgpool = nn.AdaptiveMaxPool2d((1,1))
model.fc = nn.Linear(2048, 3)
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr = 3e-4, weight_decay = 1e-5)

Loss = nn.CrossEntropyLoss()

#this will go from epoch = 1 to "epochs" inclusive
#in the for (data, lebel) in train_loader, it will know "data" refers to each image and "label" refers to folder name 
losses = []
accs = []
testlosses = []
testaccs = []
auc1s = []
auc2s = []
auc3s = []
model.best_testacc = 0
model.best_testauc = 0
model.accforbestauc = 0
trainauc1s = []
trainauc2s = []
trainauc3s = []


def plotAUC(fpr,tpr, roc_auc):
	plt.figure()
	for i in range(3):
		plt.plot(fpr[i], tpr[i], lw=2,
		label='ROC curve of grade {0} (area = {1:0.2f})'
		''.format(i+1, roc_auc[i]))
	plt.plot([0, 1], [0, 1], 'k--', lw=2)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic Curve')
	plt.legend(loc="lower right")
	plt.savefig('valROCcurve.png', dpi = 220)

def eval(): #evaluates validation set 
	with torch.no_grad():
		model.eval() #turns off batch norm and dropout and any other training only regulizers etc
		acc_sum = 0
		loss_sum = 0
		n_sum = 0
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		outs = torch.Tensor([]).to(device)
		labels = torch.Tensor([]).long().to(device)
		for q, batch in enumerate(val_loader, 1):
			data, label = batch[0].to(device), batch[1].to(device)
			out = F.softmax(model(data), dim=-1)
			outs = torch.cat((outs, out), dim = 0)
			labels = torch.cat((labels, label), dim = 0)
			loss = Loss(out, label)
			_, pred = out.max(-1)
			acc = pred.eq(label).float().mean().item()
			acc_sum += acc*data.size(0)
			loss_sum += loss.item()*data.size(0)
			n_sum += data.size(0)
		for i in range(3): #3 = number of classes, grades 1, 2, and 3 
			#fpr[i], tpr[i], _ = roc_curve(labeltest.to(device), outs)
			fpr[i], tpr[i], _ = roc_curve(label_binarize(labels.cpu().numpy(), classes=[0,1,2])[:,i], outs.cpu().numpy()[:,i])
			roc_auc[i] = auc(fpr[i], tpr[i])
	print(f'acc {acc_sum/n_sum:.3f}; loss {loss_sum/n_sum:.4f}')
	print(f'AUC class 1 {roc_auc[0]}; AUC class 2 {roc_auc[1]}; AUC class 3 {roc_auc[2]}')
	if ((roc_auc[0]+roc_auc[1]+roc_auc[2])/3 > model.best_testauc):
		model.best_testauc = (roc_auc[0]+roc_auc[1]+roc_auc[2])/3
		model.accforbestauc = acc_sum/n_sum
		plotAUC(fpr, tpr, roc_auc)
		torch.save(model, f'BestModel.pth')
		torch.save(optim, f'BestOptim.pth')
	model.train()
	return loss_sum/n_sum, acc_sum/n_sum, roc_auc[0], roc_auc[1], roc_auc[2]

def check(): #making sure AUC is evaluating right on the train set 
	with torch.no_grad():
		model.eval() #turns off batch norm and dropout and any other training only regulizers etc
		fpr_train = dict()
		tpr_train = dict()
		roc_auc_train = dict()
		outs_train = torch.Tensor([]).to(device)
		labels = torch.Tensor([]).long().to(device)
		for q, batch in enumerate(train_loader, 1):
			data, label = batch[0].to(device), batch[1].to(device)
			out = F.softmax(model(data), dim=-1)
			outs_train = torch.cat((outs_train, out), dim = 0)
			labels = torch.cat((labels, label), dim = 0)
			loss = Loss(out, label)
		for i in range(3): #3 = number of classes, grades 1, 2, and 3
			fpr_train[i], tpr_train[i], _ = roc_curve(label_binarize(labels.cpu().numpy(), classes=[0,1,2])[:,i], outs_train.cpu().numpy()[:,i])
			roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])

	print(f'Train AUC class 1 {roc_auc_train[0]}; Train AUC class 2 {roc_auc_train[1]}; Train AUC class 3 {roc_auc_train[2]}')
	model.train()
	return roc_auc_train

for epoch in range(1, epochs+1):
	acc_sum = 0
	losses_sum = 0
	n_sum = 0
	model.epoch = epoch

	for q, batch in enumerate(train_loader, 1):
		data, label = batch[0].to(device), batch[1].to(device)

		data += torch.randn(*data.shape).to(device)/256.

		out = model(data)
		loss = Loss(out, label)
		optim.zero_grad() #makes all the grads zero to start with 
		loss.backward() #backprop step, gives each of the parameters their gradients 
		optim.step()
		_, pred = out.max(-1) #returns the max value of the final dimension --> 3 class prediction
		acc = pred.eq(label.view_as(pred)).float().mean().item()
		acc_sum = acc_sum + acc*data.size(0) #data.size makes sure the averages are appropriately weighted 
		losses_sum = losses_sum + loss*data.size(0)
		n_sum = n_sum + data.size(0)

		print(f'Epoch {epoch}; batch {q}; batch acc {acc:.3f}; avg acc {acc_sum/n_sum:.3f}; avg loss {loss.item():.4f}')

		#print(data.shape, label)
	testloss,testacc, auc1, auc2, auc3 = eval()
	roc_auc_train = check()
	trainauc1s.append(roc_auc_train[0])
	trainauc2s.append(roc_auc_train[1])
	trainauc3s.append(roc_auc_train[2])
	testlosses.append(testloss)
	testaccs.append(testacc)
	auc1s.append(auc1)
	auc2s.append(auc2)
	auc3s.append(auc3)
	accs.append(acc_sum/n_sum)
	losses.append(losses_sum/n_sum)
	#if (testacc > model.best_testacc):
	#	torch.save(model, f'BestModel.pth')
	#	torch.save(optim, f'BestOptim.pth')
	#	model.best_testacc = testacc
np.save('TestAccs.npy', np.array(testaccs))
np.save('TestLosses.npy', np.array(testlosses))
np.save('testAUC1.npy', np.array(auc1s))
np.save('testAUC2.npy', np.array(auc2s))
np.save('testAUC3.npy', np.array(auc3s))
np.save('trainAUC1.npy', np.array(trainauc1s))
np.save('trainAUC2.npy', np.array(trainauc2s))
np.save('trainAUC3.npy', np.array(trainauc3s))
np.save('Losses.npy', np.array(losses))
np.save('Accuracies.npy', np.array(accs))
torch.save(model, f'Model_final.pth')




