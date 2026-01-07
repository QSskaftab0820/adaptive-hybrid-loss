# adaptive-hybrid-loss
Adaptive Hybrid Loss for Imbalanced Deep Learning

📌 Project Overview

This project implements a novel Adaptive Hybrid Loss framework for handling imbalanced classification problems in deep learning.
The approach is inspired by recent research on loss functions and dynamically adjusts loss weights during training based on validation F1-score.

The framework is demonstrated on a credit card fraud detection use case, where the minority class (fraud) is extremely underrepresented.

🚀 Key Idea

Instead of using a single static loss function, this project combines:

Cross-Entropy Loss – for stable general learning

Focal Loss – to emphasize hard-to-classify and minority samples

Dice Loss – to improve minority-class sensitivity

The weights of these loss functions are adaptively updated during training based on the model’s validation F1-score.

🧠 Novel Contribution

Dynamic loss weight adjustment using performance feedback

Adaptive focus on minority class when model performance degrades

Practical implementation inspired by recent deep learning literature

This adaptive strategy improves the balance between precision and recall in imbalanced datasets.

🗂️ Project Structure
src/

├── loss_functions.py   # Custom loss functions and adaptive hybrid loss

├── model.py            # Neural network architecture

├── train.py            # Training loop with adaptive loss logic

├── evaluate.py         # Evaluation metrics and visualization logic

├── utils.py            # Data loading and preprocessing

└── __init__.py

requirements.txt        # Python dependencies

⚙️ Requirements

Python 3.8+

TensorFlow

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

Install all dependencies using:

pip install -r requirements.txt

▶️ How to Run

Clone the repository:

git clone https://github.com/QSskaftab0820/adaptive-hybrid-loss.git

cd adaptive-hybrid-loss


(Optional) Create a virtual environment:

python -m venv venv

source venv/bin/activate   # Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Run the training script:

python src/train.py

📊 Evaluation Metrics

The model is evaluated using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Due to class imbalance, F1-score and AUC are considered the primary performance indicators.

📁 Dataset

The dataset used is a publicly available credit card fraud detection dataset.

⚠️ Dataset files are not included in this repository due to size and licensing constraints.

Please place the dataset in a local data/ directory if you wish to run the code.

📄 Related Documents

This repository supports a larger project that includes:

A detailed Research Paper (PDF)

A Case Study (PDF)

A Video Explanation

These documents were submitted separately as part of the Developer Round-1 assignment.

✍️ Author

SK AFTAB

Developer Round-1 Submission

Field: Data Science / Machine Learning

📜 License

This project is for academic and evaluation purposes only.
