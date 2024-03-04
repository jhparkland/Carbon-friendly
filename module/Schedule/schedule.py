"""
Save 파일 저장 Flag를 확인하고 Save 파일 다운로드
이후 Flag를 False로 변경(0)

Commond를 이용해서 모델 실행.
"""
import sys
import os
import time
# 현재 스크립트의 경로를 기준으로 상위 디렉토리의 경로를 sys.path에 추가합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Firebase.firebase import FirebaseManager  # 'module.' 접두사 제거
from Firebase.storage import Storage    # 'module.' 접두사 제거


def run_training():
    command = f"nvidia-smi"
    os.system(command)

def check_flag():
    """
    saved flag를 확인하고, True면 다운로드
    """
    
    firebase = FirebaseManager()
    db = firebase.db
    storage = Storage() 

    # saved_flag = firebase.read_data("optim/saved_flag") # saved_flag를 읽어옴
    # file_name = firebase.read_data("optim/file_name") # file_name을 읽어옴
    
    # 실시간 감지할 경로 설정
    db_ref =  storage.db.child("optim/file_name")
    # 리얼타임데이터베이스 실시간 감지
    db_ref.stream(storage.stream_handler)
    # 변경 사항을 계속 감지하려면 프로그램을 실행 중으로 유지


    # if saved_flag: # saved_flag가 True면
    #     storage.download_storage('KR', f'{file_name}.pt', os.getcwd()) # 저장된 파일을 다운로드 (현재 경로)
    #     firebase.write_data("optim/saved_flag", False)
    #     run_training() # 모델 실행


if __name__ == "__main__":
    check_flag()