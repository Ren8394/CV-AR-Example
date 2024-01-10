
# CV-AR_Example

---

## Dataset

Kaggle HAR Dataset: [https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones](https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones)
Please register a Kaggle account, download the dataset, and extract it to the `data` directory of this project.

After extracting the dataset, the `data` directory structure should look like this:

```plain
data
├── Human Activity Recognition
│   ├── train
│   ├── test
│   ├── Testing_set.csv
│   ├── Training_set.csv
```

## Usage

This example is run on **Python 3.11**. You can either use python virtual environment or conda environment to run this example.
After activating the environment, run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

To run the example, run the following command:
Note that the data should be downloaded and extracted to the `data` directory before running the example.

```bash
python data/HAR_parsing.py
sh run.sh
```

File explanation:

1. `run.sh`: The script to run the example. (vgg16 model is used and set as defalut neteork in this example)
2. `main.py`: The main script to run the example. It contains the following training, validation, and testing steps.
3. `results_vis.ipynb`: This file is used to easily visualize the results of the example.
4. `data/HAR_parsing.py`: This file is used to parse, preprocess, and split the dataset.
5. `dataset/HAR.py`: This file is used to define the dataset class. You can see the code to understand how the dataset is defined, and how the data is loaded.

After running the example, the `results` and `weights` directory structure should be created and look like this:

```plain
results
├── pred_class_count_{model}.txt
├── true_class_count_{model}.txt
├── train_loss_{model}.txt
├── val_loss_{model}.txt
```

```plain
weights
├── best_{model}.pth
```

## Results

You can execute the `results_vis.ipynb` file to visualize the results of the example.
