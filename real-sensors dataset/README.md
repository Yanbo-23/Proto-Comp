## Dataset Description
This dataset folder contains a total of 1251 .pcd files, divided into 30 different categories.
The data can be downloaded through [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/117019ce0d884b1c9646/?dl=1) or [Google Drive](https://drive.google.com/file/d/1qtDG3Poo5S_VbrgfDdaf3bT2ORP0qCXk/view?usp=drive_link). 
The data format is as follows.
├── bathtub/

│   ├── bathtub_000.pcd      

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

## Application Method Description

### Step 1: Point Cloud Completion
1. **Model Application**: Prepare a pre-trained ProtoComp model to complete the real_sensors benchmark point clouds.
2. **Preprocessing**: Before completion, apply Farthest Point Sampling (FPS) and perform centralization on the point clouds to prepare them for model input.
3. **Completion**: Input the preprocessed point clouds and corresponding prompts with the format of 'A ${category}.$' into the completion model to generate the completed point clouds.
4. **Fidelity Metric Calculation**: Calculate the Fidelity metric by comparing the completed point clouds directly with the input point clouds.

### Instructions

To run the test script, use the following command:

```bash
bash ./scripts/test.sh ${GPU_NAME} --ckpts ${CKPTS_PATH} --config ${CONFIG_PATH} --save_path ${SAVE_PATH}
```

#### Parameters:

- **${GPU_NAME}**:  specifies the name or number of the GPU to use.

- **${CKPTS_PATH}**:  specifies the path to the model checkpoint file that contains the model parameters saved at a specific training epoch.

- **${CONFIG_PATH}**: specifies the path to the configuration file, which contains settings needed during training or testing.

- **${SAVE_PATH}**: specifies the path where the results will be saved. 



### Step 2: Point Cloud Classification
1. **Data Loading**: Use a DataLoader to load the predicted point clouds generated in Step 1.
2. **Model Application**: Use the officially trained PointNet++ model to classify all the predicted point clouds.
3. **Geometric Discriminability Metric Calculation**: Compare the classification results with their true labels and then calculate the Geometric Discriminability metric based on the comparison results.

### Instructions
```bash
python test_classification.py --log_dir ${LOG_PATH} --save_path ${SAVE_PATH}
```
#### Parameters:

- **${LOG_PATH}**:  specifies the directory path which contains the model checkpoint.

- **${SAVE_PATH}**:  specifies the directory path containing the results predicted by ProtoComp.


## Additional Notes
- Ensure all dependencies are installed, including libraries for point cloud processing and machine learning frameworks compatible with the pre-trained models.
- Refer to the [official documentation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git) for detailed instructions on how to use the PointNet++ model.

