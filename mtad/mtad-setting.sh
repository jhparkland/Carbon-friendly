echo "System Update"
apt update # 시스템 업데이트

# 파이썬 설치 및 필요한 프로그램 설치
echo "Install python 3.8 and wget"
apt-get install python3.8 wget 
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# PiP 설치
echo "Install pip"
wget https://bootstrap.pypa.io/get-pip.py
python3.8 get-pip.py
 
# 파이토치 설치
echo "Install torch 201+cu118"
pip install -r /mtad-gat-pytorch/requirements.txt
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118