![model-demo](https://github.com/user-attachments/assets/2bb56633-b869-4aca-82db-3d82f60b0c7a)

# Food-101 MobileNet Fine-Tuning

This project focuses on fine-tuning a MobileNet model on the Food-101 dataset, aiming to optimise model performance through experimentation while addressing the challenges of limited computational resources. Given these constraints, we conducted our experiments and hyperparameter optimisation on a 10% subset of the training data. Two distinct fine-tuning strategies were explored: single-stage fine-tuning, where the entire model was trained simultaneously, and two-stage fine-tuning, which involved an initial phase of training only the final Fully-Connected layer followed by a phase of training the entire network. The two-stage approach aimed to leverage rapid adaptation of the classifier before fine-tuning the pre-trained layers with more nuanced adjustments. Our results demonstrated that, of the experimental models, single-stage fine-tuning achieved the highest accuracy at 67.3%, while two-stage fine-tuning achieved 66.1%. Ultimately, we applied single-stage fine-tuning (without freezing) for the final model which we trained on the entire dataset, resulting in a validation accuracy of 83.5% and a test accuracy of 82.9%.

This repository includes a [Notebook](https://github.com/gordon801/mobile-net-food-101/blob/main/mobile-net-food-101.ipynb) that summarises our methodologies and results, providing insights into the fine-tuning process. It also includes a [Flask web application](https://github.com/gordon801/mobile-net-food-101/blob/main/app.py) that deploys the trained final model, which predicts one of the 101 food classes in the Food-101 dataset. This allows users to upload their own images and receive predictions on its food class.

## Architecture
![Project Architecture](https://github.com/user-attachments/assets/fdd81f0c-94ce-44be-8890-cbbd9b79da10)
This project uses MobileNetV3 Large, which we fine-tune on the Food-101 dataset by replacing the final fully-connected layer to output predictions for the 101 food classes. The model is trained using cross-entropy loss and its performance is evaluated using accuracy.

## Experimental Model Performance
![Validation performance](https://github.com/user-attachments/assets/22a4b296-0c7c-4792-b775-890cd9ab7976)
- Single-Stage Fine-tuning: 67.3% val acc
- Single-Stage Fine-tuning with freezing (i.e. training only the classifier): 53.0% val acc
- Two-Stage Fine-tuning v1 (with overfit classifier): 63.9% val acc
- Two-Stage Fine-tuning v2 (without overfitting issue): 66.1% val acc

## Final Model Outputs
![Prediction examples](https://github.com/user-attachments/assets/90b2eb85-e733-4441-ac50-0c1d7e4ed77b)

## Data
This model was trained and tested on the FOOD-101 dataset. More information about this dataset can be found on [this website](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). The dataset can be downloaded from [here](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz).

## TODO
- Steps to reproduce 
- Other details
- References/Acknowledgements
- Full report
