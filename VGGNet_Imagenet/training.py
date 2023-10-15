import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import scipy
from tqdm import tqdm 

from vggmodel import VGGNet
import torch.optim as optim

import sys
import os

# import gpu_utils modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import gpu_utils
import multiprocessing
from multiprocessing.managers import BaseManager

class MyManager(BaseManager):   pass
MyManager.register('Check_GPU', gpu_utils.Check_GPU)

# GPU info measurement modules
manager = MyManager()




def train():
    # For dataset preprocessing. Not for transformer model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    # dataset load
    # !!!! Need to change dataset PATH!!!!!!!
    print('train set load...')
    trainset = torchvision.datasets.ImageNet('D:/data', split='train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    print('valid set load...')
    validset = torchvision.datasets.ImageNet('D:/data', split='val', transform=transform)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=False)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    ## cuda setting
    # select device
    if torch.cuda.is_available():
        device = torch.device('cuda')  
        print('Using gpu device')
    else:
        device = torch.device('cpu')
        print('Using cpu device')

    ## cuDNN disable(when crush occurred) 
    # torch.backends.cudnn.enabled = False  


    ## creating model and cuda connecting
    net = VGGNet()
    net = net.to(device)
    param = list(net.parameters())

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)


    print('training start')
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0

        if(epoch>0):
            net = VGGNet()
            net.load_state_dict(torch.load(save_path))
            net.to(device)

        i = 0
        # Iteration 
        for data in tqdm(trainloader, desc='Processing'):
            # gpu measurement start of Iteration
            iter_measurement_start = multiprocessing.Process(target=gpu_measure.iter_start, args=(epoch, i, trainloader.batch_size))
            iter_measurement_start.start()

            # # for optimizing core frequency
            # setting_clock = multiprocessing.Process(target=gpu_measure.set_gpu_core_clock, args=(optimzied_core_clock,))
            # setting_clock.start()

            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs,f = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if(loss.item() > 1000):
                print(loss.item())
                for param in net.parameters():
                    print(param.data)

            
            # print statistics
            running_loss += loss.item()
            
            torch.cuda.synchronize()

            # # iter gpu measurement & log save
            iter_measurement_end = multiprocessing.Process(target=gpu_measure.iter_end, args=())
            iter_measurement_end.start()
            
            if i % 100 == 99:    # print every 100 iteration
                print('%d epoch, %5d iter | loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

                # save gpu measurement result
                measure_save = multiprocessing.Process(target=gpu_measure.save_csv, args=())
                measure_save.start()
                measure_save.join()

            i += 1
            

        save_path="ILSVRC_VGGNet/Experiment/exp/" + epoch + "_model_states.pth"
        torch.save(net.state_dict(), save_path)
        
        print(f'{epoch}\'epoch model saved at', save_path, '...')
        

    print('Finished Training')
    

    class_correct = list(0. for i in range(1000))
    class_total = list(0. for i in range(1000))
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs,_ = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(16):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy_sum=0
    # for i in range(1000):
    #     temp = 100 * class_correct[i] / class_total[i]
    #     print('Accuracy of %5s : %2d %%' % (
    #         idx2label[i], temp))
    #     accuracy_sum+=temp
    print('Accuracy(all class): ', accuracy_sum/1000)




if __name__ == '__main__':
    # # if error occurred in Windows
    multiprocessing.freeze_support()
    manager.start()
    gpu_measure = manager.Check_GPU()
    try:
        # training start
        train()
        
    # for setting gpu's default frequency when terminated training.py process by KeyboardInterrupt
    except KeyboardInterrupt:
        gpu_measure.stop = True
        gpu_measure.reset_gpu_core_clock()
        sys.exit()
