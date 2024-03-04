"""
firebase storage와 연동

동작 정의
모델 파라미터 파일이 업로드 -> 업로드 이벤트 감지 -> 모델 파라미터 전이

test
임의 파일 업로드 -> 업로드 이벤트 감지 -> 업로드된 파일 메타 데이터 출력
"""


import os,sys
import subprocess
import random
# 현재 스크립트의 경로를 기준으로 상위 디렉토리의 경로를 sys.path에 추가합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Firebase.firebase import FirebaseManager  # 'module.' 접두사 제거

class Storage:
    def __init__(self) -> None:
        self.fm = FirebaseManager() # firebase 계정 연결
        self.db = self.fm.db
        self.storage = self.fm.app.storage() # storage 연결  

    # 스토리지 파일 쓰기
    def write_storage (self,location, file_name, file_auth):
        # 스토리지 파일 추가
        self.storage.child(location).child(file_name).put(file_auth) 
    
    # 스토리지 파일 URL 가져오기
    def get_storage (self, location, file_name):
        url = self.storage.child(location).child(file_name).get_url(None)
        return url

    # 스토리지 파일 다운로드
    def download_storage (self, location, file_name, download_path):
        current_dir = os.getcwd() # 현재 경로 저장
        os.chdir(download_path) #다운할경로로 변경
        self.storage.child(location).child(file_name).download(path=download_path, filename=file_name) # 파일을 읽어와서 저장                                                                                                   # 저장 경로의 파일을 저장
        os.chdir(current_dir) # 경로 복구             
                                                                                 
    # Realtime Database 변경 사항을 감지하는 이벤트 핸들러
    def stream_handler(self, message):
        # print("Received a change in the Realtime Database:")
        if message["event"] == "put":  
            if message["data"] is None: # 데이터 삭제시 반응
                print("Data deleted:", message["path"])
            else:  # 데이터 삽입, 변경시 반응
                print("Data added or updated:", message["data"])
                # 데이터 삽입,변경 감지시  스토리지 파일을 저장
                storage_file = message["data"] # 저장할 파일 이름 가져오기
                download_path = "/workspace/home/dsl2023/learning-passage" # 저장할 경로 설정
                if storage_file != '':
                    self.download_storage("main", storage_file, download_path)
                    self.fm.write_data("optim/saved_flag", False)
                    self.fm.write_data("optim/file_name", '')
                    zone = 'FR'
                    print(f"현재 지역: {zone}")
                    self.fm.write_data("main/zone", zone)
                    self.fm.write_data("optim/changed", True)
                    
        elif message["event"] == "patch":  # 데이터 부분 업데이트시 반응
            print("Data updated:", message["data"])                                                                                                       