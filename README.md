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

