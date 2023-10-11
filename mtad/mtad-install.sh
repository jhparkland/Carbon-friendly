echo "Set Current path permission"
sudo chmod 777 -R .

echo "Git Clone"
sudo git clone https://github.com/ML4ITS/mtad-gat-pytorch.git

echo "Docker image load(mtad)"
sudo sudo docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04

echo "Get docker images"
sudo docker images

echo "docker container create by mtad image"
sudo docker run -it --gpus all --name mtad -v /:/app nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04
