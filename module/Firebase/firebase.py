import pyrebase

class FirebaseManager:
    """
    firebase 연결 매니저
    """

    def __init__(self):
        # Firebase database init
        self.config = {
                       
                    }
        self.app = pyrebase.initialize_app(self.config) # firebase app에 대한 참조 가져오기
        self.db = self.app.database() # realtime database 참조 가져오기
        self.auth = self.app.auth() # auth 서비스에 대해 참조 가져오기

    #Realtime Database 데이터 삽입
    def write_data(self, location, data):
        # 지정된 위치에 데이터 쓰기 set(덮어쓰기) , push는 추가
        self.db.child(location).set(data)


    # Realtime Database 데이터 읽어오기
    def read_data(self, location):
        all_data = self.db.child(location).get() # 데이터읽어오기
        if all_data is not None:
            return all_data.val() # 데이터 내용 반환
        else:
            return []  # 데이터가 없는 경우 빈 리스트 반환


    # Realtime Database 변경 사항을 감지하는 이벤트 핸들러
    def stream_handler(self, message):
        print("Received a change in the Realtime Database:")
        if message["event"] == "put":  
            if message["data"] is None: # 데이터 삭제시 반응
                print("Data deleted:", message["path"])
            else:  # 데이터 삽입, 변경시 반응
                print("Data added or updated:", message["data"])
        elif message["event"] == "patch":  # 데이터 부분 업데이트시 반응
            print("Data updated:", message["data"]) 