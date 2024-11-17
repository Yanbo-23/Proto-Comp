## Real Sensors Dataset Description
This dataset folder contains a total of 1251 .pcd files, divided into 30 different categories. The data can be downloaded through [[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/f/076097900274447cb3bd/?dl=1) or [[Google Drive]](https://drive.google.com/file/d/1OzQg1-_GefA8NOVa8h8Y4oPoouUHerVI/view?usp=sharing). 
The data format is as follows:
```
├── bathtub/
│   ├──bathtub_000.pcd      
│   ├── bathtub_001.pcd       
│   ├── ...
│   └── bathtub_049.pcd
├── bed/
│   ├── bed_000.pcd
│   ├── bed_001.pcd
│   ├── ...
│   └── bed_049.pcd
├── ...
├── wardrobe/
│   └── wardrobe_000.pcd
```
Then put the downloaded folder to data/real_sensors_benchmark.
## Application Method Description

### Step 1: Point Cloud Completion

To run the test script, use the following command:

```bash
bash ./scripts/test.sh ${GPU_NAME} --ckpts ${CKPTS_PATH} --config ${CONFIG_PATH} --save_name ${SAVE_NAME}
```
Then the completed points will be saved and the **Fidelity** metric will be calculated.

#### Parameters:

- ```GPU_NAME```:  specifies the name or number of the GPU to use.

- ```CKPTS_PATH```:  specifies the path to the model checkpoint file that contains the model parameters saved at a specific training epoch.

- ```CONFIG_PATH```: specifies the path to the configuration file, which contains settings needed during training or testing.

- ```SAVE_NAME```: specifies the path name where the results will be saved. 
#### Example
```
bash ./scripts/test.sh 0 --ckpts experiments/ProtoComp/PCN_models/example/pcn.pth --config cfgs/PCN_models/ProtoComp.yaml --save_path test
```

### Step 2: Point Cloud Classification

The you can calculate **Geometric Discriminability** metric by running:  
```bash
cd ./Pointnet_Pointnet2_pytorch
python test_classification.py --log_dir ${LOG_PATH} --data_path ${DATA_PATH}
```
#### Parameters:

- ```${LOG_PATH}```:  specifies the directory path which contains the (PointNet++) model checkpoint.

- ```${DATA_PATH}```:  specifies the directory path containing the results predicted by ProtoComp.
#### Example
```
python test_classification.py --log_dir pointnet2_cls_ssg --data_path ../data/real_sensors_benchmark
```
## Additional Notes
- Ensure all dependencies are installed, including libraries for point cloud processing and machine learning frameworks compatible with the pre-trained models.
- Refer to the [official documentation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git) for detailed instructions on how to use the PointNet++ model.
