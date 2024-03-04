import platform
import psutil
import pynvml
import sys, os
import csv
import pandas as pd

class resource_info:

    def __init__(self):
        self.cpu_name = 0
        self.cpu_curr_use = 0
        self.gpu_name = 0
        self.gpu_curr_use = 0
        self.ram_size = 0
        self.ram_curr_use = 0

    def cpuName(self):
        self.self.cpu_name = platform.processor()
        return self.cpu_name

    def cpuCurrUse(self) : # cpu 현재 사용량을 가져오는 함수.
        self.cpu_use = str(psutil.cpu_percent()) + "%"

        return self.cpu_use # return type ; String 

    def gpuName(self) : # gpu 장치 이름을 가져오는 함수. 
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = pynvml.nvmlDeviceGetName(handle)
            pynvml.nvmlShutdown()
            return self.gpu_name
        except Exception as e:
            return str(e)

    def gpuCurrUse(self):
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
             # 총 GPU 메모리와 현재 사용 중인 GPU 메모리를 MB 단위로 계산
            total_memory_mb = memory_info.total / (1024 ** 2)
            used_memory_mb = memory_info.used / (1024 ** 2)

            # 현재 사용 중인 메모리의 백분율을 계산
            self.gpu_curr_use = str(round((used_memory_mb / total_memory_mb) * 100, 3))
            print(self.gpu_curr_use)
            pynvml.nvmlShutdown()

            return self.gpu_curr_use
        except Exception as e:
            print(f"GPU 사용량 측정 실패: {e}")
            return -1  # 또는 오류를 나타내는 다른 방식 사용

    def ramSize(self) :
        self.ram_size = str(round(psutil.virtual_memory().total / (1024.0 **3)))+"(GB)"

        return self.ram_size # return type ; String 

    def ramCurrUse(self) : # RAM 현재 사용량을 가져오는 함수. 
        pid = os.getpid()
        curr_process = psutil.Process(pid)
        memory_use_mb = curr_process.memory_info()[0] / 2**20
        total_memory_mb = psutil.virtual_memory().total / 2**20

        self.ram_curr_use = round((memory_use_mb / total_memory_mb) * 100, 3)
        return self.ram_curr_use # return type ; float




