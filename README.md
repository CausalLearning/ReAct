# ReAct: Temporal Action Detection with Relational Queries

---
This repo holds the code for React, which is accept to ECCV2022. If you have any question, welcome to contact at "shidingfeng at buaa . edu. cn".


## Installation
We build our code based on the MMaction2 project (1.3.10 version). See [here](https://github.com/open-mmlab/mmaction2) for more details if you are interested.
MMCV is needed before install MMaction2, which can be install with:
```shell
pip install mmcv-full-f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
# For example, to install the latest mmcv-full with CUDA 11.1 and PyTorch 1.9.0, use the following command:
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

```
For other CUDA or pytorch version, please refer [here](https://github.com/open-mmlab/mmcv) to get a matched link. 


Then, our code can be built by 
```shell
git clone https://github.com/sssste/React.git
cd React
pip3 install -e .
```

Then, Install the 1D Grid Sampling and RoI Align operators. 
```shell
cd React/model
python setup.py build_ext --inplace
```

## Data preparing 
We used the TSN feature [(Google Drive Link)](https://drive.google.com/drive/folders/1-19PgCRTTNfy2RWGErvUUlT0_3J-qEb8) provied by G-TAD for our model. Please put all the **files** in the ```datasets/thumos14/``` fold (or you can put them in any place and modify the data path in the config file in ```React/configs/thumos_tsn_feature.py```)

## Training

Our model can be trained with

```python
python tools/train.py React/configs/thumos_tsn_feature.py --validate 
```

We recommend to set the `--validate` flag to monitor the training process.
 
## Test
If you want to test the pretrained model, please use the following code.
```shell
python tools/test.py React/configs/thumos_tsn_feature.py PATH_TO_MODEL_PARAMETER_FILE
```

We provide the pretrained weights for React ([THUMOS14](https://drive.google.com/file/d/1pcfJ6G5SC_zNeWG11cxtXAhj2rJVrz8q/view?usp=sharing)) . Our code supports test with a batch of videos for efficient. If you want to change the batch size, you can change the number of ```workers_per_gpu``` in ```thumos_tsn_feature.py```. 

Then, you can run the test by 
```shell
python tools/test.py React/configs/thumos_tsn_feature.py react_thumos_pretrained_weight.pth
```

The results (mAP at tIoUs, %) should be 

| Method | 0.3  |  0.4 | 0.5 |0.6 | 0.7| Avg|
|--------|------|-----|-----|-----|-----|-----|
| React  | 70.8 |65.9|57.8|47.2|34.2|55.2

## Citation
If you feel this work useful, please cite our paper! Thank you!
```
@inproceedings{shi2022react,
  title={ReAct: Temporal Action Detection with Relational Queries},
  author={Dingfeng Shi, Yujie Zhong, Qiong Cao, Jing Zhang, Lin Ma, Jia Li and Dacheng Tao},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

