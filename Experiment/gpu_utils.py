import time
import subprocess
import pandas as pd
import numpy as np
import os
import pyrebase
from scipy.optimize import minimize


class Check_GPU:
    def __init__(self, model_name, gpu_id, voltage_rate) -> None:
        # DL model name
        self.dl_model = model_name
        # GPU device ID 
        self.gpu_id = gpu_id
        # undervolting clock rate
        self.under_volting_rate = voltage_rate

        ## Optimzing Section
        # cost function weights(between execution_time and power_usage)
        self.cost_weight = {'execution_time':0.05, 'power_usage':0.75, 'freq_scaling':0.2}
        self.sampling_size = 10
        

        # idle gpu power & workload
        self.default_gpu_usage = None
        self.default_gpu_usage = self.get_gpu_usage()


        # total consumtion, total execute time
        self.total_excution_time = 0.0
        self.total_energy_usage = 0.0

        # gpu min, max frequency (core, memory)
        self.freq_minmax_info = self.get_min_max_freq_info()

        self.cfreq_min = self.freq_minmax_info['MinCoreClock']
        self.cfreq_max = self.freq_minmax_info['MaxCoreClock']

        # setable frequency list
        self.cfreq_list = self.freq_minmax_info['AvailCoreClock']

        # to change frequency for undervolting experiment
        if self.under_volting_rate != None:
            self.uv_freq = self.cfreq_max * self.under_volting_rate

        self.epoch = 0
        self.iter = 0
        self.batch_size = 0
        self.iter_execution_time = 0.0
        self.iter_energy_usage = 0.0
        self.workload = None
        self.core_freq = self.get_gpu_freq_info()['CoreClock']
        self.prev_opt_core_freq = self.core_freq
        self.opt_core_freq = None

        # GPU info 
        self.device_info = self.get_gpu_info()
        print(self.device_info)
        if self.freq_minmax_info:
            print(self.freq_minmax_info)
        print('Default gpu freq :', self.get_gpu_freq_info())
        # setting frequency (not use optimization algorithm(L-BFGS))
        if self.under_volting_rate != None:
            self.set_gpu_core_clock(int(self.uv_freq))

        self.cur_df = pd.DataFrame(columns=['DeviceID', 'DLmodel', 'TimeStamp', 'EpcohIdx', 'IterIdx', 'ExecutionTime', 'Energy', 'Workload', 'ExecutionTimePerData', 'EnergyPerData', 'CoreFreq', 'OtimalCoreFreq'])
        if self.under_volting_rate != None:
            os.makedirs(f'/workspace/home/dsl2023/learning-passage/exp{int(self.under_volting_rate*100)}', exist_ok=True)
        else:
            os.makedirs(f'/workspace/home/dsl2023/learning-passage/exp', exist_ok=True)

        # firebase config
        db_config = {
            "apiKey": "AIzaSyCIhOSHqDgjbe9LU2x45xByd8g4Y2P18HM",
            "authDomain": "carbon-friendly-402901.firebaseapp.com",
            "databaseURL": "https://carbon-friendly-402901-default-rtdb.firebaseio.com",
            "projectId": "carbon-friendly-402901",
            "storageBucket": "carbon-friendly-402901.appspot.com",
            "messagingSenderId": "982587361472",
            "appId": "1:982587361472:web:3c7e267e476bad79674525"
        }
        
        
        # Pyrebase initialization
        firebase = pyrebase.initialize_app(db_config)
        # Real-time Database
        self.db = firebase.database()
        # Storage
        self.storage = firebase.storage()



        
    def get_last_process(self, queue):
        idx = self.db.child('main').child('lastProcess').get().val()
        queue.put(idx)


    def get_zone(self):
        self.zone = self.db.child('main').child('zone').get().val()
        print('zone', self.zone)
        return self.zone
        

    def get_save_flag(self, result_queue):
        self.save_flag = self.db.child('optim').child('request').get().val()
        if self.save_flag == True:
            print('save_flag :', self.save_flag)
        result_queue.put(self.save_flag)

    
    def get_gpu_id(self, queue=None):
        self.gpu_id = self.db.child('main').child('gpuId').get().val()
        
        if queue != None:
            queue.put(self.gpu_id)
        else:
            return self.gpu_id


    def set_gpu_id(self):
        id = self.get_gpu_id()
        self.db.child('main').child('gpuId').set(id)

    
    def save_model_firebase(self, model_path):
        # save firebase storage
        filename = self.dl_model + "_" + str(self.iter) + '_iter_model.pt'
        self.storage.child('main/'+filename).put(model_path)                    
        self.db.child('optim').child('request').set(False)
        self.db.child('optim').child('saved_flag').set(True)
        self.db.child('optim').child('file_name').set(filename)
        self.db.child('main').child('lastProcess').set(self.iter)

    
    def load_model_from_firebase(self):
        # load model
        zone = self.db.child('main').child('lastProcess').get().val()
        i = self.db.child('main').child('lastProcess').get().val()
        model_path = "/workspace/home/dsl2023/learning-passage/" + self.dl_model + "_" + str(i) +"_iter_model.pt"
        file_name = self.dl_model + "_" + str(i) + '_iter_model.pt'
        self.storage.child('main/'+file_name).download(model_path)
        print(f'load_{i}th iter\'s model...')

        self.zone = zone


    # GPU 전력 사용량 측정 함수
    def get_gpu_usage(self):
        # sudo nvidia-smi command
        command = f"sudo nvidia-smi --id {self.gpu_id} --query-gpu=power.draw --format=csv,noheader,nounits"
        result = subprocess.check_output(command, shell=True, universal_newlines=True)

        # result parsing
        power_draw = float(result.strip())  # Convert to kW

        if self.default_gpu_usage != None:
            power_draw -= self.default_gpu_usage
        
        if power_draw > 0:
            return power_draw
        else:
            return 0.0
    
    # # GPU 작업량 측정 함수
    # def get_gpu_workload(self):
    #     # sudo nvidia-smi command
    #     command = f'sudo nvidia-smi --id {self.gpu_id} --query-gpu=memory.used --format=csv,noheader,nounits'
    #     output = subprocess.check_output(command, shell=True)
        
    #     if self.default_gpu_workload is not None:
    #         output -= self.default_gpu_workload

    #     # output units : MB
    #     if output > 0:
    #         return output
    #     else:
    #         return 0




    # GPU 이름 가져오기
    def get_gpu_info(self):
        try:
            # sudo nvidia-smi 명령 실행
            command = f"sudo nvidia-smi --id {self.gpu_id} --query-gpu=name,memory.total --format=csv,noheader,nounits"
            result = subprocess.check_output(command, shell=True, universal_newlines=True)

            # 결과 파싱
            lines = result.strip().split('\n')

            gpu_name, memory_total = lines[0].strip().split(',')
            
            return {
                "GPU Name": gpu_name.strip(),
                "Memory Total": memory_total.strip(),
            }
        except Exception as e:
            print("Error:", e)
            return None



    # GPU 최소, 최대 주파수 정보 가져오기
    def get_min_max_freq_info(self):
        try:
            # "sudo nvidia-smi -q -d SUPPORTED_CLOCKS" 명령 실행
            command = f"sudo nvidia-smi --id {self.gpu_id} -q -d SUPPORTED_CLOCKS"
            result = subprocess.check_output(command, shell=True, universal_newlines=True)

            # 결과 문자열에서 "Graphics" 주파수 정보 추출
            graphics_frequencies = []

            # 결과 문자열을 줄 단위로 분할
            lines = result.strip().split('\n')
            in_graphics_section = False
            cnt = 0

            for line in lines:
                # "Supported Clocks" 섹션 진입 여부 확인
                if "Supported Clocks" in line:
                    in_graphics_section = True
                elif in_graphics_section:
                    # "Memory" 주파수 정보가 나오면 루프 종료
                    if "Memory" in line:
                        cnt += 1
                        if cnt > 1:
                            break
                    # "Graphics" 주파수 정보 추출
                    elif "Graphics" in line:
                        # 주파수 값에서 'MHz' 문자를 제거하고 숫자로 변환하여 저장
                        frequency_str = line.split(":")[1].strip().replace('MHz', '')
                        frequency = float(frequency_str)
                        graphics_frequencies.append(frequency)

            # # 추출된 "Graphics" 주파수 리스트 출력
            # print("Graphics Frequencies:", graphics_frequencies)
            return {'MinCoreClock': min(graphics_frequencies), 
                    'MaxCoreClock': max(graphics_frequencies),
                    'AvailCoreClock': graphics_frequencies,
                    }

        except Exception as e:
            print("#Error#:", e)
            return None

    # set {gpu_index} gpu's core clock to {core_clock_freq}
    def set_gpu_core_clock(self, core_clock_freq):
        try:
            # persistent mode on
            command = f"sudo nvidia-smi --id {self.gpu_id} -pm=1"
            subprocess.run(command, shell=True, check=True)

            # setting command for change gpu core clock
            command = f"sudo nvidia-smi --id {self.gpu_id} --lock-gpu-clocks={core_clock_freq},{core_clock_freq}"
            # sudo nvidia-smi 명령 실행
            subprocess.run(command, shell=True, check=True)
            cur_clock = self.get_gpu_freq_info()['CoreClock']

            return cur_clock

        except subprocess.CalledProcessError as e:
            print("Error:", e)
            return None



    def reset_gpu_core_clock(self):
        try:
            # reset command of gpu core clock
            command = f"sudo nvidia-smi -i {self.gpu_id} --reset-gpu-clocks"

            # sudo nvidia-smi command execute
            subprocess.run(command, shell=True, check=True)

            time.sleep(1000)
            cur_clock = self.get_gpu_freq_info()['CoreClock']

            print(f"Core Clock reset to {cur_clock['CoreClock']} MHz ")
            
            return cur_clock

        except subprocess.CalledProcessError as e:
            print("Error:", e)
            return None



    # 현재 gpu 주파수 측정
    def get_gpu_freq_info(self):
        try:
            # Getting GPU info by sudo nvidia-smi query
            command = f"sudo nvidia-smi --id {self.gpu_id} --query-gpu=clocks.current.memory,clocks.current.graphics --format=csv,noheader,nounits"
            result = subprocess.check_output(command, shell=True, universal_newlines=True)

            # 결과 파싱
            lines = result.strip().split('\n')
            memory_clock, core_clock = map(int, lines[0].strip().split(','))

            return {
                "MemoryClock": memory_clock,
                "CoreClock": core_clock,
            }
        except Exception as e:
            print("Error:", e)
            return None


    # 매 iter 시작마다 호출
    def iter_start(self, e, i, b_size, is_eval=False):
        # epoch, iter info allocate
        self.epoch = e
        self.iter = i
        self.batch_size = b_size
            
        # variable value initailization for current iteration
        self.iter_start_time = time.time()
        self.iter_energy_usage = 0.0
        self.core_freq = 0
        # self.opt_core_freq = 0
        self.stop = False
        
        prev_time = time.time()
        # 1 while := 0.015s
        while not self.stop:
            cur_time = time.time()
            exec_time = cur_time - prev_time

            # power_usage measurement
            # W * (second / 3600) -> kWh
            self.iter_energy_usage += self.get_gpu_usage() * (exec_time / 3600.0)

            prev_time = cur_time
            
            

        
        

    ### append dictionary data of iteration measurement to dataframe
    def iter_end(self, workload):
        self.iter_execution_time = time.time() - self.iter_start_time
        self.core_freq = self.get_gpu_freq_info()['CoreClock']

        # optimizing frequency
        if workload != None:
            # self.prof.stop()
            # events = self.prof.key_averages()
            # data = [{k: v for k, v in e.__dict__.items()} for e in events]
            # df = pd.DataFrame(data)
            self.workload = workload
            self.opt_core_freq = self.optimize()
            self.set_gpu_core_clock(self.opt_core_freq)

        # timestamp
        self.stop = True
        timestamp = time.strftime('%Y-%m-%d %I:%M:%S %p', time.localtime(time.time()))
        self.total_excution_time += self.iter_execution_time
        self.total_energy_usage += self.iter_energy_usage
        # freq_info = self.get_gpu_freq_info()
        

        row = {'DeviceID': self.device_info['GPU Name'],
            'DLmodel': self.dl_model,
            'TimeStamp': timestamp,
            'EpcohIdx': self.epoch,
            'IterIdx': self.iter,
            'ExecutionTime': self.iter_execution_time,
            'Energy': self.iter_energy_usage,
            'Workload': self.workload,
            'ExecutionTimePerData': self.iter_execution_time / self.batch_size,
            'EnergyPerData': self.iter_energy_usage / self.batch_size,
            'CoreFreq': self.core_freq,
            'TotalExecutionTime': self.total_excution_time,
            'TotalEnergy': self.total_energy_usage,
            # 아직 최적화 알고리즘 구현 X
            'OtimalCoreFreq':self.opt_core_freq}
        
        db_data = {
            'coreFreq' : self.core_freq,
            'executionTime': self.iter_execution_time,
            'energyUsage' : self.iter_energy_usage
        }

        # send core_freq to firebase
        self.db.child('optim').child('coreFreq').set(db_data['coreFreq'])
        self.db.child('optim').child('energyUsage').set(db_data['energyUsage'])
        self.db.child('optim').child('executionTime').set(db_data['executionTime'])

        # print dictionary data of iteration
        print(f"iter measurement:{row}")

        self.cur_df.loc[len(self.cur_df) + 1] = row

    def save_csv(self):
        # append measurement to total csv
        if self.under_volting_rate != None:
            save_path = f"/workspace/home/dsl2023/learning-passage/exp{int(self.under_volting_rate * 100)}/{int(self.under_volting_rate * 100)}_measurement.csv"
        else:
            save_path = f"/workspace/home/dsl2023/learning-passage/exp/optim_measurement.csv"

        try:
            total_df = pd.read_csv(save_path)
        except Exception:
            total_df = pd.DataFrame(columns=['DeviceID', 'DLmodel', 'TimeStamp', 'EpcohIdx', 'IterIdx', 'ExecutionTime', 'Energy', 'Workload', 'ExecutionTimePerData', 'EnergyPerData', 'CoreFreq', 'OtimalCoreFreq'])
        total_df = pd.concat([total_df, self.cur_df])

        # save csv
        total_df.to_csv(save_path, index=False)
        print(f'measurement saved into ', save_path)

        # for next save
        total_df = None
        self.cur_df = pd.DataFrame(columns=['DeviceID', 'DLmodel', 'TimeStamp', 'EpcohIdx', 'IterIdx', 'ExecutionTime', 'Energy', 'Workload', 'ExecutionTimePerData', 'EnergyPerData', 'CoreFreq', 'OtimalCoreFreq'])


    def optimize(self):
        result = minimize(fun=self.cost_func, 
                                x0=np.array(self.core_freq, dtype=float), 
                                bounds=self.constraint_bounds(), 
                                method='L-BFGS-B'
                                )
        optimal_freq = int(result.x[0])
        return optimal_freq

    def cost_func(self, f):
        # freq, workload is numpy.array of (h,) shape (sampleing interval)

        # t = time_const * workload / freq
        # p = time_const * workload * freq^3
        exec_t = self.cur_df['ExecutionTime'].tail(self.sampling_size).to_numpy()
        power_c = self.cur_df['Energy'].tail(self.sampling_size).to_numpy()
        work_l = self.cur_df['Workload'].tail(self.sampling_size).to_numpy()
        core_f = self.cur_df['CoreFreq'].tail(self.sampling_size).to_numpy()
        opt_f = self.cur_df['OtimalCoreFreq'].tail(self.sampling_size).to_numpy()

        time_constant = self.get_time_constant(exec_t, work_l, core_f)
        power_constant = self.get_power_constant(power_c, work_l, core_f) * 10
        scaling_constant = 1.0

        time_cost = time_constant * work_l / f
        power_cost = power_constant * work_l * (f ** 3)
        scaling_cost = scaling_constant * (opt_f - f)**2
        
        result = sum(self.cost_weight['execution_time'] * time_cost) + sum(self.cost_weight['power_usage'] * power_cost) + sum(self.cost_weight['freq_scaling'] * scaling_cost)

        return float(result)
    
    
    def constraint_bounds(self):
        bounds = self.get_min_max_freq_info()
        min = bounds['MinCoreClock']
        max = bounds['MaxCoreClock']
        return [(min, max)]

    # to model time cost function
    def get_time_constant(self, exec_time, workload, freq):
        return exec_time * freq / workload

    # to model power cost function
    def get_power_constant(self, power, workload, freq):
        return power / workload / (freq**3)
