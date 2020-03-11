import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from torchvision import datasets, transforms
import pdb
from sklearn.metrics import roc_curve, auc

epochs = 5
K = 5

#loading the data in, batch size = minibatch size
#important to have the data in its own folder with labels in diff folders as is done here
train_loader = torch.utils.data.DataLoader(
datasets.ImageFolder("data",
transform = transforms.Compose([transforms.Resize((256,256)),transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]) #reshape to 512x512 b/c they were unequal and they're all around 512
),
batch_size = 100, shuffle = True 
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

images = torch.Tensor([])
labels = torch.Tensor([]).long()
for q, (data,label) in enumerate(train_loader, 1):
	print("Gathering Batch", q)
	images = torch.cat((images, data), dim = 0)
	labels = torch.cat((labels, label), dim = 0)
groups = torch.chunk(torch.randperm(len(images)), 5)


def train_and_test_set(data, labels, groups, k, device):

	# Specific
	spec_xtrain = torch.Tensor([])
	spec_strain = torch.Tensor([]).long()
	for j in range(k):
		if j != (k-1):
			spec_xtrain = torch.cat((spec_xtrain, data[groups[j]]))
			spec_strain = torch.cat((spec_strain, labels[groups[j]]))
		else: 
			spec_xtest = data[groups[j]]
			spec_stest = labels[groups[j]]

	spec_xtrain = spec_xtrain.to(device)
	spec_strain = spec_strain.to(device)
	spec_xtest = spec_xtest.to(device)
	spec_stest = spec_stest.to(device)
	return spec_xtrain, spec_strain, spec_xtest, spec_stest

datatrain, labeltrain, datatest, labeltest = train_and_test_set(images, labels, groups, K, device)





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

model = torch.hub.load('pytorch/vision:v0.5.0', 'resnext101_32x8d', pretrained=True).to(device)
model.fc = nn.Linear(2048, 3)
model.avgpool = nn.AdaptiveMaxPool2d((1,1))
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr = 3e-4, weight_decay = 1e-4)

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
#for epoch in range(1, epochs+1):
#	acc_sum = 0
#	losses_sum = 0
#	n_sum = 0
#	for q, (data, label) in enumerate(train_loader,1):
#
#		data = data.to(device)
#		target = target.to(device)
#		out = model(data)
#		loss = Loss(out, label)
#		optim.zero_grad() #makes all the grads zero to start with 
#		loss.backward() #backprop step, gives each of the parameters their gradients 
#		optim.step()
#		_, pred = out.max(-1)
#		acc = pred.eq(label.view_as(pred)).float().mean().item()
#		acc_sum = acc_sum + acc*data.size(0) #data.size makes sure the averages are appropriately weighted 
#		losses_sum = losses_sum + loss*data.size(0)
#		n_sum = n_sum + data.size(0)
#
#		print(f'Epoch {epoch}; batch {q}; batch acc {acc:.3f}; avg acc {acc_sum/n_sum:.3f}; avg loss {loss.item():.4f}')
#
#		#print(data.shape, label)
#
#	accs.append(acc_sum/n_sum)
#	losses.append(losses_sum/n_sum)
#	torch.save(model, f'Model_{epoch:03d}.pth')
#	torch.save(optim, f'Optim_{epoch:03d}.pth')
#np.save('Losses.npy', np.array(losses))
#np.save('Accuracies.npy', np.array(accs))
#torch.save(model, f'Model_final.pth')
def plotAUC(fpr,tpr, roc_auc):
	plt.figure()
	for i in range(3):
		plt.plot(fpr[i], tpr[i], lw=4,
		label='ROC curve of grade {0} (area = {1:0.2f})'
		''.format(i+1, roc_auc[i]))
	plt.plot([0, 1], [0, 1], 'k--', lw=4)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic Curve')
	plt.legend(loc="lower right")
	plt.savefig('ROCcurve.png', dpi = 220)

def eval(): #evaluates test set 
	with torch.no_grad():
		model.eval() #turns off batch norm and dropout and any other training only regulizers etc
		batches = torch.chunk(torch.randperm(len(datatest)), 20) #100 batches
		acc_sum = 0
		loss_sum = 0
		n_sum = 0
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		outs = torch.Tensor([]).to(device)
		for q, batch in enumerate(batches, 1):
			out = F.softmax(model(datatest[batch].to(device)), dim=-1)
			outs = torch.cat((out, outs), dim = 0)
			loss = Loss(out, labeltest[batch].to(device))
			_, pred = out.max(-1)
			acc = pred.eq(labeltest[batch].view_as(pred)).float().mean().item()
			acc_sum += acc*datatest[batch].size(0)
			loss_sum += loss.item()*datatest[batch].size(0)
			n_sum += datatest[batch].size(0)
		for i in range(3): #3 = number of classes, grades 1, 2, and 3 
			#fpr[i], tpr[i], _ = roc_curve(labeltest.to(device), outs)
			fpr[i], tpr[i], _ = roc_curve(label_binarize(labeltest.cpu().numpy(), classes=[0,1,2])[:,i], outs.cpu().numpy()[:,i])
			roc_auc[i] = auc(fpr[i], tpr[i])
	print(f'acc {acc_sum/n_sum:.3f}; loss {loss_sum/n_sum:.4f}')
	print(f'AUC class 1 {roc_auc[0]}; AUC class 2 {roc_auc[1]}; AUC class 3 {roc_auc[2]}')
	model.train()
	if ((roc_auc[0]+roc_auc[1]+roc_auc[2])/3 > model.best_testauc):
		model.best_testauc = (roc_auc[0]+roc_auc[1]+roc_auc[2])/3
		model.accforbestauc = acc_sum/n_sum
		plotAUC(fpr, tpr, roc_auc)
		torch.save(model, f'BestModel.pth')
		torch.save(optim, f'BestOptim.pth')
	return loss_sum/n_sum, acc_sum/n_sum, roc_auc[0], roc_auc[1], roc_auc[2]

for epoch in range(1, epochs+1):
	acc_sum = 0
	losses_sum = 0
	n_sum = 0
	batches = torch.chunk(torch.randperm(len(datatrain)), 69) #100 batches 
	model.epoch = epoch

	for q, batch in enumerate(batches, 1):

		data = datatrain[batch].to(device)
		target = labeltrain[batch].to(device)
		out = model(data)
		loss = Loss(out, target)
		optim.zero_grad() #makes all the grads zero to start with 
		loss.backward() #backprop step, gives each of the parameters their gradients 
		optim.step()
		_, pred = out.max(-1) #returns the max value of the final dimension --> 3 class prediction
		acc = pred.eq(target.view_as(pred)).float().mean().item()
		acc_sum = acc_sum + acc*data.size(0) #data.size makes sure the averages are appropriately weighted 
		losses_sum = losses_sum + loss*data.size(0)
		n_sum = n_sum + data.size(0)

		print(f'Epoch {epoch}; batch {q}; batch acc {acc:.3f}; avg acc {acc_sum/n_sum:.3f}; avg loss {loss.item():.4f}')

		#print(data.shape, label)
	testloss,testacc, auc1, auc2, auc3 = eval()
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
np.save('AUC1.npy', np.array(auc1s))
np.save('AUC2.npy', np.array(auc2s))
np.save('AUC3.npy', np.array(auc3s))
np.save('Losses.npy', np.array(losses))
np.save('Accuracies.npy', np.array(accs))
torch.save(model, f'Model_final.pth')




