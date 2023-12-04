import csv
import pandas as pd
import sys
from module.Firebase.firebase_manager import FirebaseManager
from module.Firebase.storage_manager  import Storage

#CSV 파일 경로 설정
#csv_file="data/100_measurement.csv"
#CSV 파일을 pandas 데이터프레임으로 읽어오기
#df = pd.read_csv(csv_file)

# 원하는 컬럼만 선택해서 읽어오기 
#ExecutionTime = [round(value, 5) for value in df["ExecutionTime"].to_list()]
#Energy = df["Energy"].to_list()
#ExecutionTimePerData = [round(value, 5) for value in df["ExecutionTimePerData"].to_list()]

# FirebaseManager 인스턴스 생성
firebase_manager = FirebaseManager()

# 데이터를 Firebase Realtime Database에 쓰기
# firebase_manager.write_data("main/ExecutionTime", ExecutionTime)
# firebase_manager.write_data("main/Energy", Energy )
# firebase_manager.write_data("main/ExecutionTimePerData", ExecutionTimePerData)
# firebase_manager.write_data("FR/ev", 100)
# firebase_manager.write_data("FR/emission", 280)
# firebase_manager.write_data("FR/preemission", 0)
# firebase_manager.write_data("FR/cmpemission", 0)
# firebase_manager.write_data("FR/zone", 'FR')
# firebase_manager.write_data("FR/gfreq", 700)
# firebase_manager.write_data("FR/preintensity", 0)
# firebase_manager.write_data("FR/cmpintensity", 0)
# firebase_manager.write_data("DE/preev", 0)
# firebase_manager.write_data("DE/cmpev", 0)
# firebase_manager.write_data("DE/pregfreq", 0)
# firebase_manager.write_data("DE/cmpgfreq", 0)

# 리얼타임데이터베이스에서 읽어오기 
# read_all_ExecutionTime = firebase_manager.read_data("main/ExecutionTime")
# read_all_Energy = firebase_manager.read_data("main/Energy")
# read_all_ExecutionTimePerData = firebase_manager.read_data("main/ExecutionTimePerData")
# 읽어온 데이터 출력
#for read_ExecutionTime in read_all_ExecutionTime:
#    print(read_ExecutionTime)
#for read_Energy in read_all_Energy:
#    print(read_Energy)
#for read_ExecutionTimePerData in read_all_ExecutionTimePerData:
#    print(read_ExecutionTimePerData)


# 스토리지 인스턴스 생성
storage_manager = Storage()

# 스토리지 파일 추가
#file_path = "assets/pt/model.pt"  # 해당 파일 경로
#storage_manager.write_storage("JP-TK", "model.pt", file_path) # 스토리지 삽입
#firebase_manager.write_data("JP-TK/FileName", "model.pt") # 리얼타임데이터베이스에 파일이름 삽입

# 리얼타임데이터베이스에서 파일이름 가져오기
zone = firebase_manager.read_data("main/zone")
storage_file = firebase_manager.read_data(f"{zone}/FileName")

# 스토리지 파일 url 가져오기
#url = storage_manager.get_storage(f"{zone}",storage_file)
#print(url)

# 데이터베이스 변경사항 실시간 감지
# 실시간 감지할 경로 설정
db_ref =  storage_manager.db.child(f"{zone}/FileName")
# 리얼타임데이터베이스 실시간 감지
db_ref.stream(storage_manager.stream_handler)
# 변경 사항을 계속 감지하려면 프로그램을 실행 중으로 유지
input("Press Ctrl+C to exit...")
