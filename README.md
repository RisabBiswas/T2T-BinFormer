# 🥇SOTA Document Image Enhancement - A Layer-Wise Tokens-to-Token Transformer Network for Improved Historical Document Image Enhancement
The official PyTorch code for the project [A Layer-Wise Tokens-to-Token Transformer Network for Improved Historical Document Image Enhancement](https://arxiv.org/abs/2312.03946).

## Description
We propose to employ a Tokens-to-Token Transformer network for document image enhancement, a novel encoder-decoder architecture based on a tokens-to-token vision transformer.

![alt text](Architecture.png?raw=true)

![alt text](Result_Comparison.png?raw=true)

## Step 1 - Download Code
Clone the repository to your desired location:
```bash
git clone https://github.com/RisabBiswas/T2T-BinFormer
cd T2T-BinFormer
```
## Step 2 - Process Data
### Data Path
The research and experiments are conducted on the DIBCO and H-DIBCO datasets. Find the dataset here - [Link](https://drive.google.com/drive/folders/1u8vDqRlxWe5GvRPr6cD-C7GeL9MSqBsX?usp=drive_link). After downloading, extract the folder named DIBCOSETS and place it in your desired data path. 
Means:  /YOUR_DATA_PATH/DIBCOSETS/

### Additional Data Path
* PALM Dataset - [Link](https://drive.google.com/drive/folders/1u8vDqRlxWe5GvRPr6cD-C7GeL9MSqBsX?usp=drive_link)
* Persian Heritage Image Binarization Dataset - [Link](https://drive.google.com/drive/folders/1CqP_2t7jBb9mqe4hjLJ_JDwd8vEUkyM9?usp=drive_link)
* Degraded Maps - [Link](https://drive.google.com/drive/folders/1Li2x0pHfkmwx0kVXoj4kJ7DQuaZt83GO?usp=sharing)

### Data Splitting
Specify the data path, split size, validation, and testing sets to prepare your data. In this example, we set the split size as (256 X 256), the validation set as 2016, and the testing set as 2018 while running the process_dibco.py file.
 
```bash
python process_dibco.py --data_path /YOUR_DATA_PATH/ --split_size 256 --testing_dataset 2018 --validation_dataset 2016
```

## Using T2T-BinFormer
### Step 3 - Training
For training, specify the desired settings (batch_size, patch_size, model_size, split_size, and training epochs) when running the file train.py. For example, for a base model with a patch size of (16 X 16) and a batch size of 32, we use the following command:

```bash
python train.py --data_path /YOUR_DATA_PATH/ --batch_size 32 --vit_model_size base --vit_patch_size 16 --epochs 151 --split_size 256 --validation_dataset 2016
```
You will get visualization results from the validation dataset on each epoch in a folder named vis+"YOUR_EXPERIMENT_SETTINGS" (it will be created). In the previous case, it will be named visbase_256_16. Also, the best weights will be saved in the folder named "weights".
 
### Step 4 - Testing on a DIBCO dataset
To test the trained model on a specific DIBCO dataset (should match the one specified in Section Process Data, if not, run process_dibco.py again). Use your own trained model weights. Then, run the below command. Here, I test on H-DIBCO 2017, using the base model with a 16X16 patch size and a batch size of 16. The binarized images will be in the folder ./vis+"YOUR_CONFIGS_HERE"/epoch_testing/ 
```bash
python test.py --data_path /YOUR_DATA_PATH/ --model_weights_path  /THE_MODEL_WEIGHTS_PATH/  --batch_size 16 --vit_model_size base --vit_patch_size 16 --split_size 256 --testing_dataset 2017
```

## Resuts
The results of our model can be found [Here](https://drive.google.com/drive/folders/1LojmH8AfAumZDWoQOLRikWXpYYgfF6TL?usp=sharing).

## Acknowledgement
Our project has adapted and borrowed the code structure from [DocEnTr](https://github.com/dali92002/DocEnTR/tree/main). We are thankful to the authors! Additionally, we really appreciate the great work done on [vit_pytorch](https://github.com/lucidrains/vit-pytorch/tree/main) by [Phil Wang](https://github.com/lucidrains).

## Authors
- [Risab Biswas](https://www.linkedin.com/in/risab-biswas/)
- [Swalpa Kumar Roy](https://swalpa.github.io/)
- [Umapada Pal](https://www.isical.ac.in/~umapada/)


## Citation

If you use the T2T-BinFormer code in your research, we would appreciate a citation to the original paper:
```
  @misc{biswas2023layerwise,
        title={A Layer-Wise Tokens-to-Token Transformer Network for Improved Historical Document Image Enhancement}, 
        author={Risab Biswas and Swalpa Kumar Roy and Umapada Pal},
        year={2023},
        eprint={2312.03946},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
  }
```

## Contact 
If you have any questions, please feel free to reach out to <a href="mailto:risabbiswas19@gmail.com" target="_blank">Risab Biswas</a>.


## Conclusion
We really appreciate your interest in our research. The code should not have any bugs, but if there are any, we are really sorry about that. Do let us know in the issues section, and we will fix it ASAP! Cheers! 
