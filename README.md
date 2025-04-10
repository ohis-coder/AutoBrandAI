# Car Brand Prediction with Neural Network

## Overview

This project is a machine learning-based car brand prediction system built using a neural network. The model takes user input, such as car model names or keywords, and predicts the corresponding car brand (e.g., Honda, Toyota, Ford). The neural network is trained on a dataset of popular Nigerian car brands and their associated car models. The goal is to identify the car brand based on the car's attributes mentioned in a user-provided statement.

## Features

- **Prediction of Car Brand**: The system predicts the car brand (e.g., Honda, Toyota) based on input keywords such as car model names or terms related to the car.
- **Neural Network Model**: A simple neural network with a hidden layer, ReLU activation, and softmax output layer.
- **Confidence Threshold**: The model provides predictions with a confidence score, and if the confidence is below a specified threshold, it returns "Unknown".
- **Interactive Input**: Users can enter a statement about their car, and the model will predict the brand, providing a user-friendly way to interact with the system.
  
## Technologies Used

- **Python**: The core programming language for the system.
- **Numpy**: Used for matrix operations and implementing the neural network layers.
- **Regular Expressions (re)**: Used for text tokenization to extract keywords from user input.

## Setup and Installation

### Prerequisites

- Python 3.x
- Numpy library (can be installed via pip)

To install Numpy, run:

```bash
pip install numpy
```

### Running the Project

1. Clone or download the repository.
2. Navigate to the project directory.
3. Run the script using Python:

```bash
python car_brand_prediction.py
```

4. Enter a statement about a car model when prompted. For example, you can input "this expedition is fast" to predict the brand.
5. The system will predict the car brand based on the input.

### Example Usage

When prompted, you can enter a car-related statement like:

```
Enter statement (or type 'exit' to quit): this expedition is fast
Predicted Brand: Ford
```

### Exiting the Program

Type `exit` to quit the program.

## How It Works

1. **Tokenizer**: The input text is tokenized into words using regular expressions (`re.findall`), and each word is converted to lowercase for uniformity.
   
2. **Data Preparation**: The car brands and associated keywords/models are pre-processed. Each brand and its models/terms are mapped into a list of tokens.

3. **Vectorization**: The words in the dataset are converted into one-hot encoded vectors, where each unique word in the vocabulary is represented by a vector with 1's and 0's. This vectorization represents the input data for the neural network.

4. **Neural Network**:
   - **Forward Pass**: The neural network computes the output through the hidden layer using matrix multiplications, applying ReLU activation, and passing the result through a softmax function for classification.
   - **Backpropagation**: The model uses gradient descent to adjust weights and biases by calculating the error and updating the parameters accordingly.
   - **Training**: The network is trained for 1000 epochs (iterations) with a learning rate of 0.1.

5. **Prediction**: After training, the model can predict the car brand from a user-provided input by passing it through the trained network and interpreting the output probabilities.

## Neural Network Architecture

- **Input Layer**: Size of the vocabulary.
- **Hidden Layer**: 10 neurons, with ReLU activation.
- **Output Layer**: One neuron per car brand, with softmax activation to provide probability distribution over car brands.

## Customizing the Model

- **Add More Brands**: To add more car brands, simply update the `car_brands` dictionary with new brand names and their associated models.
- **Modify Hidden Layer Size**: Adjust the `hidden_size` parameter in the `NeuralNetwork` class to increase or decrease the number of neurons in the hidden layer.
- **Adjust Training**: Modify the `epochs` and `learning_rate` parameters to improve the model's performance or speed.

## Contributing

Feel free to fork this repository and submit pull requests for improvements or additional features. If you find any bugs or issues, please open an issue to discuss them.

## License

This project is open-source and available 
