
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ ! -e $DIR/data ]]; then
    mkdir "$DIR/data"
else
    echo "'data' directory already exists."
fi

gdown -O "$DIR/data/traffic_analysis.mov" "https://drive.google.com/uc?id=1qadBd7lgpediafCpL_yedGjQPk-FLK-W"
gdown -O "$DIR/data/traffic_analysis.pt" "https://drive.google.com/uc?id=1y-IfToCjRXa3ZdC1JpnKRopC7mcQW-5z"