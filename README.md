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

## File Structure

The main directory should contain the following scripts:  
- `layers.py`  
- `model.py`  
- `data_utils.py`  
- `train.py`  

### Multiple Versions of `data_utils.py` and `train.py`

These scripts are available in different versions, indicated by the following suffixes:  
- **`ds`**: Accelerated by DeepSeed  
- **`dp`**: Accelerated by DataParallel  
- **`ddp`**: Accelerated by DistributedDataParallel  

### Running DDP Mode

To run scripts in **DDP mode**, execute commands like `train_ddp.sh` in the terminal.  

### Example Output

The output of the script will be similar to `retrieval_results.json`.  



主目录下应同时有layers.py, model.py, data_utils.py, 和 train.py 四个脚本。

其中 data_utils 和 train 有多个对应版本。后缀ds表示采用deepseed加速，dp表示采用DataParallel加速，ddp表示采用DistributeDataParallel加速。

