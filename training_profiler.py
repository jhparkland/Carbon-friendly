import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import profiler
import scipy
from tqdm import tqdm 

from vggmodel import VGGNet
import torch.optim as optim

import pandas as pd
import sys
import os

## import gpu_utils modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import gpu_utils_profiler
import multiprocessing
from multiprocessing.managers import BaseManager
# # if error occurred in Windows
torch.multiprocessing.freeze_support()
class MyManager(BaseManager):   pass
MyManager.register('Check_GPU', gpu_utils_profiler.Check_GPU)
global gpu_id




def train(is_suspend = False):
    # For dataset preprocessing. Not for transformer model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # dataset load
    # !!!! Need to change dataset PATH!!!!!!!
    trainset = torchvision.datasets.ImageNet('D://data', split='train', transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    
    validset = torchvision.datasets.ImageNet('D://data', split='val', transform=transform)
    validloader = DataLoader(validset, batch_size=64, shuffle=True)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ## cuda setting
    # select device
    if torch.cuda.is_available():
        # gpu select
        device = torch.device(f'cuda:{gpu_id}')  
        print('Using gpu device')
    else:
        device = torch.device('cpu')
        print('Using cpu device')

    ## cuDNN disable(when crush occurred) 
    # torch.backends.cudnn.enabled = False  

    prof = profiler._KinetoProfile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA], profile_memory=True, record_shapes=True)

    ## creating model and cuda connecting
    net = VGGNet()
    net = net.to(device)
    param = list(net.parameters())

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.00001)
    
    start_i_idx = 0
    if is_suspend == True:
        queue = multiprocessing.Queue()
        find_last_process = multiprocessing.Process(target=gpu_measure.get_last_process, args=(queue,))
        load = multiprocessing.Process(target=gpu_measure.load_model_from_firebase, args=())
        find_last_process.start()
        load.start()
        find_last_process.join()
        load.join()
        start_i_idx = queue.get()

        model_path="/Experiment/exp/" + model_name + "latest_model.pt"
        state_dict = torch.load(model_path)
        # state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        net.load_state_dict(state_dict)
        net.to(device)


    for epoch in range(3):  # loop over the dataset multiple times
        running_loss = 0.0

        train_iter = iter(trainloader)
        
        i = 1
        # Iteration 
        for data in tqdm(train_iter, desc='Processing'):
            if i > start_i_idx:
                # gpu measurement start of Iteration
                iter_measurement_start = multiprocessing.Process(target=gpu_measure.iter_start, args=(epoch, i, trainloader.batch_size,))
                iter_measurement_start.start()
                
                if i % 10 == 1:
                    prof.start()

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
                if i % 10 == 1:
                    prof.stop()
                    df = pd.DataFrame([{k: v for k, v in e.__dict__.items()} for e in prof.key_averages()])
                    workload = df[df['cuda_memory_usage']>0]['cuda_memory_usage'].sum()
                    # 기록 초기화를 위해 새로운 객체 생성
                    prof = profiler._KinetoProfile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA], profile_memory=True, record_shapes=True)

                    iter_measurement_end = multiprocessing.Process(target=gpu_measure.iter_end, args=(workload,))
                else:
                    iter_measurement_end = multiprocessing.Process(target=gpu_measure.iter_end, args=(None,))

                iter_measurement_end.start()

                
                if i % 100 == 0:    # print every 100 iteration
                    print('%d epoch, %5d iter | loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

                    # save gpu measurement result
                    measure_save = multiprocessing.Process(target=gpu_measure.save_csv, args=())
                    measure_save.start()
                    measure_save.join()

                queue = multiprocessing.Queue()
                zone = multiprocessing.Process(target=gpu_measure.get_zone, args=())
                save_flag = multiprocessing.Process(target=gpu_measure.get_save_flag, args=(queue,))

                zone.start()
                save_flag.start()
                save_flag.join()
                s_flag = queue.get()

                if s_flag == 1:
                    if voltage_rate != None:
                        model_path="/Experiment/exp" + model_name + str(int(voltage_rate * 100)) + "/" + str(i) + "_iter_model.pt"
                    else:
                        model_path="/Experiment/exp/" + model_name + str(i) + "_iter_model.pt"
                    
                    # model.pt save
                    torch.save(net.state_dict(), model_path)
                    
                    # save model.pt to firebase storage
                    save_model = multiprocessing.Process(target=gpu_measure.save_model_firebase, args=(model_path,))
                    save_model.start()

                    # save i'th iter model states
                    print(f'{i} iter model saved at', model_path, '...')
                
            i += 1
            

    print('Finished Training')
    

    class_correct = list(0. for i in range(1000))
    class_total = list(0. for i in range(1000))
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
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
    
    
    ###################################################
    # Experiment condition setting
    voltage_rate = None
    gpu_id = 0
    model_name = 'VGGNet'
    is_train_from_first = True
    ###################################################

    gpu_measure = manager.Check_GPU(model_name, gpu_id, voltage_rate)

    # start_i_idx = 0
    # if voltage_rate != None:
    #     root_dir = '/Experiment/exp' + str(int(voltage_rate * 100))
    # else:
    #     root_dir = '/Experiment/exp'

    

    # os.makedirs(root_dir, exist_ok=True)
    # for (root, dirs, files) in os.walk(root_dir):
    #     if len(files) > 0 and not is_train_from_first:
    #         model_list = [int(m.split(sep='_')[0]) for m in files]
    #         start_i_idx = max(model_list)
    #     else:
    #         start_i_idx = 0

    try:
        # training start
        train()
        gpu_measure.stop = True
        gpu_measure.reset_gpu_core_clock()
        
    # for setting gpu's default frequency when terminated training.py process by KeyboardInterrupt
    except KeyboardInterrupt | Exception:
        gpu_measure.stop = True
        gpu_measure.reset_gpu_core_clock()
        sys.exit()
