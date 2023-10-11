model=$1
dataset=$2
epochs=$3


if [[ "$model" == "mtad" ]];  then
    echo "Selected Model = "$model
    python3 "../../mtad-gat-pytorch/train.py" --dataset $dataset --epochs $epochs

elif [[ "$model" == "VggNet" ]]; then
    echo "Selected Model = $model"
    # python3 "./"

elif [[ "$model" == "ResNet" ]]; then
    echo "Selected Model = $model"

else
    echo "Not define model" 
fi
