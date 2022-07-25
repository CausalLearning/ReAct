# ReAct: Temporal Action Detection with Relational Queries

---
This repo holds the code for React, which is accept to ECCV2022.
## Installation
First install the MMCV. See [here](https://github.com/open-mmlab/mmcv) for more details.
```shell
pip install mmcv-full=={mmcv_version} -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

We build our code based on the MMaction2 project (1.3.10 version). See [here](https://github.com/open-mmlab/mmaction2) for more details if you are interested. Our code can be built by 
```shell
cd mmaction2
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

We provide the pretrained weights for React ([THUMOS14](https://drive.google.com/file/d/1pcfJ6G5SC_zNeWG11cxtXAhj2rJVrz8q/view?usp=sharing)) . Our code supports test with a batch of videos for efficient. If you want to change the batch size, you change the number of ```workers_per_gpu``` in ```thumos_tsn_feature.py```. 

Then, you can run the test by 
```shell
python tools/test.py React/configs/thumos_tsn_feature.py react_thumos_pretrained_weight.pth
```

The results (mAP at tIoUs, %) should be 

| Method | 0.3  |  0.4 | 0.5 |0.6 | 0.7| Avg|
|--------|------|-----|-----|-----|-----|-----|
| React  | 70.8 |65.9|57.8|47.2|34.2|55.2


