

## Gender Recognition
Gender recognition is a salient application where you need to identify the 
gender of a person based on their face picture. In this repository, a custom made
convolutional neural network has been used to identify gender. Furthermore, you can 
integrate this classifier with detector for second stage classification.
Furthermore, you can train the classifier on your own dataset.





## Installation

```
git clone https://github.com/faizan1234567/Image-Classification.git
cd Image-Classification
   ```
Create a virtual environment on linux to keep everything about the repository
```
python3 -m venv classification
source classification/bin/activate
   ```
And for Windows.
```
python3 -m venv classification
 .\classification\Scripts\activate

   ```
Now install the required packages by running the following command
```
pip install --upgrade pip
pip install -r requirements.txt
   ```

## Usage
To train on your custom dataset, run the training command

```
python classifier.py -h
python classifier.py --epochs 200 --img 224 --data "data_path" \
 --batch 32 --workers 8 --classes 2
   ```

And, to test your model performance use the following commands

```
python test_classifier.py -h
python test_classifier.py --weights weights_path.pth --speed \
 --test_data test_set_path --all_classes_accuracy --img 224 --batch 16 \
  --workers 8 --display_samples
 ```
