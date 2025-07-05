# Handwritten Devanagari Character Recognition using CNN

This project implements a Convolutional Neural Network (CNN) based system for recognizing handwritten Devanagari characters. The system is capable of classifying 46 different Devanagari characters commonly used in languages like Hindi, Marathi, and Sanskrit.

## Project Overview

This project uses deep learning techniques to recognize handwritten Devanagari characters. The implementation is built using TensorFlow/Keras and includes:
- Data preprocessing and augmentation
- A deep CNN architecture
- Model checkpointing for best performance
- Visualization of training metrics

## Dataset

The project uses a custom dataset of handwritten Devanagari characters. The dataset is split into training and testing sets:
- Training set: For model training
- Testing set: For model evaluation

Images are processed to 32x32 grayscale format with appropriate data augmentation techniques applied during training.

## Model Architecture

The CNN architecture consists of:
1. Input layer: 32x32 grayscale images
2. Four convolutional blocks, each containing:
   - Conv2D layer with 3x3 kernel
   - Batch Normalization
   - MaxPooling layer with 2x2 pool size
3. Dense layers:
   - First dense layer with 128 units
   - Second dense layer with 64 units
   - Output layer with 46 units (one for each character class)

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib (for visualization)

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install tensorflow numpy matplotlib
```

## Usage

1. Ensure the dataset is properly organized in the following structure:
```
Hindi_Dataset/
├── Train/
│   ├── Character_1/
│   ├── Character_2/
│   └── ...
└── Test/
    ├── Character_1/
    ├── Character_2/
    └── ...
```

2. Run the training script:
```bash
python Hindi.py
```

The model will be saved as `HindiModel2.keras` with the best validation accuracy.

## Model Performance

The model uses:
- Adam optimizer
- Categorical cross-entropy loss function
- Accuracy as the evaluation metric
- 25 epochs of training
- Batch size of 32

## Data Augmentation

The training data is augmented with:
- Rotation up to 5 degrees
- Width and height shifts of 0.1
- Shear range of 0.2
- Zoom range of 0.2
- Horizontal flips disabled

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
