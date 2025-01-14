## Set up

conda create --name cir python=3.9

conda activate cir

conda install --yes -c pytorch pytorch=2.5.1 torchvision cudatoolkit=11.0

pip install ftfy regex tqdm

pip install git+https://github.com/openai/CLIP.git

change the CLIP class in file CLIP/clip.py like this:

        # return logits_per_image, logits_per_text
        
        return image_features, text_features
        

## Data Preparation

Please follow instructions: https://github.com/google-deepmind/magiclens/tree/main/data

## File Explanation

主目录下应同时有layers.py, model.py, data_utils.py, 和 train.py 四个脚本。
其中 data_utils 和 train 有多个版本。后缀ds表示采用

