import torch
from torch.utils.data import DataLoader
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
# # if error occurred in Windows
torch.multiprocessing.freeze_support()
class MyManager(BaseManager):   pass
MyManager.register('Check_GPU', gpu_utils.Check_GPU)




def train(start_i_idx):
    # For dataset preprocessing. Not for transformer model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    skip_idx = start_i_idx * 64
    # dataset load
    # !!!! Need to change dataset PATH!!!!!!!
    trainset = torchvision.datasets.ImageNet('D:/data', split='train', transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=False)
    
    validset = torchvision.datasets.ImageNet('D:/data', split='val', transform=transform)
    validloader = DataLoader(validset, batch_size=64, shuffle=False)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    ## cuda setting
    # select device
    if torch.cuda.is_available():
        # gpu select
        device = torch.device('cuda:0')  
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


    print(f'training start at {start_i_idx}th iter')
    for epoch in range(1):  # loop over the dataset multiple times
        running_loss = 0.0

        train_iter = iter(trainloader)
        i = 1
        if start_i_idx > 0:
            net = VGGNet()
            model_path="VGGNet_Imagenet/Experiment/exp" + str(int(voltage_rate * 100)) + "/" + str(start_i_idx) + "_iter_model_states.pth"
            state_dict = torch.load(model_path)
            # state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            net.load_state_dict(state_dict)
            net.to(device)
            print(f'Load {start_i_idx}\'th iter model states ')
            print('Loading dataloader for skip trained data. Please Wait.')
            for _ in range(start_i_idx):
                next(train_iter, None)
            i = start_i_idx + 1
        
        # Iteration 
        for data in tqdm(train_iter, desc='Processing'):
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
            
            if i % 100 == 0:    # print every 100 iteration
                print('%d epoch, %5d iter | loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

                
                model_path="VGGNet_Imagenet/Experiment/exp" + str(int(voltage_rate * 100)) + "/" + str(i) + "_iter_model_states.pth"
                torch.save(net.state_dict(), model_path)
                
                # save i'th iter model states
                print(f'{i} iter model saved at', model_path, '...')

                # save gpu measurement result
                measure_save = multiprocessing.Process(target=gpu_measure.save_csv, args=())
                measure_save.start()
                measure_save.join()
            i += 1
            

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
    # GPU info measurement modules
    manager = MyManager()
    manager.start()
    gpu_measure = manager.Check_GPU()

    # Experiment condition
    voltage_rate = 1.0
    
    start_i_idx = 0
    root_dir = 'VGGNet_Imagenet/Experiment/exp' + str(int(voltage_rate * 100))
    os.makedirs(root_dir, exist_ok=True)
    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            model_list = [int(m.split(sep='_')[0]) for m in files]
            start_i_idx = max(model_list)
        else:
            start_i_idx = 0

    try:
        # training start
        train(start_i_idx)
        
    # for setting gpu's default frequency when terminated training.py process by KeyboardInterrupt
    except Exception:
        gpu_measure.stop = True
        gpu_measure.reset_gpu_core_clock()
        manager.close()
        sys.exit()
