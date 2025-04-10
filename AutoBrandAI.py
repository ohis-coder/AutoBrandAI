import re
import numpy as np

# Define Nigerian car brands with their associated car models/terms
car_brands = {
    "Honda": ["CR-V", "civic", "accord", "honda", "pilot", "fit"],
    "Toyota": ["Camry", "Corolla", "landcruiser", "rav4", "toyota", "highlander"],
    "Lexus": ["RX", "Lexus", "es ", "is ", "gs ", "lx ", "nx "],
    "Mercedes-Benz": ["E350", "mercedes", "benz", "s-class", "e-class", "g-class"],
    "BMW": ["bmw", "x5", "x6", "320i", "530i", "m3"],
    "Ford": ["ford", "focus", "fiesta", "mustang", "expedition", "f-150"]
}

# Tokenizer function
def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return words

# Prepare the data
def prepare_data(car_brands):
    data = []
    labels = []
    for brand, words in car_brands.items():
        for word in words:
            data.append(tokenize(word))
            labels.append(brand)
    return data, labels

data, labels = prepare_data(car_brands)
vocabulary = list(set(word for sentence in data for word in sentence))
vocab_size = len(vocabulary)
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Vectorize the data
def vectorize_data(data, word_to_index):
    vectors = np.zeros((len(data), len(word_to_index)))
    for i, sentence in enumerate(data):
        for word in sentence:
            if word in word_to_index:
                vectors[i, word_to_index[word]] = 1
    return vectors

# One-hot encoding for labels
def one_hot_encode_labels(labels, car_brands):
    label_to_index = {brand: i for i, brand in enumerate(car_brands)}
    one_hot_labels = np.zeros((len(labels), len(car_brands)))
    for i, label in enumerate(labels):
        one_hot_labels[i, label_to_index[label]] = 1
    return one_hot_labels

X = vectorize_data(data, word_to_index)
y = one_hot_encode_labels(labels, car_brands)

# Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.hidden = np.dot(X, self.W1) + self.b1
        self.hidden = np.maximum(0, self.hidden)  # ReLU activation

        self.output = np.dot(self.hidden, self.W2) + self.b2
        self.output = self.softmax(self.output)  # Softmax for output layer
        return self.output

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def backpropagate(self, X, y, learning_rate=0.01):
        m = X.shape[0]

        output_error = self.output - y
        dW2 = np.dot(self.hidden.T, output_error) / m
        db2 = np.sum(output_error, axis=0) / m

        hidden_error = np.dot(output_error, self.W2.T) * (self.hidden > 0)
        dW1 = np.dot(X.T, hidden_error) / m
        db1 = np.sum(hidden_error, axis=0) / m

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            self.forward(X)
            self.backpropagate(X, y, learning_rate)

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Test function with confidence threshold
def test_model(nn, input_text, threshold=0.5):
    input_vector = vectorize_data([tokenize(input_text)], word_to_index)
    output = nn.forward(input_vector)
    
    # Get the index of the highest probability
    predicted_index = np.argmax(output, axis=1)[0]
    predicted_brand = list(car_brands.keys())[predicted_index]
    confidence = output[0, predicted_index]
    
    # If the confidence is below the threshold, return "Unknown"
    if confidence < threshold:
        return "Unknown"
    
    return predicted_brand

# Function to handle user input and predict brand
def predict_from_input():
    print("\nEnter a statement about your car to predict the brand!")
    while True:
        user_input = input("Enter statement (or type 'exit' to quit): ")
        
        if user_input.lower() == "exit":
            print("Exiting the program...")
            break
        
        predicted_brand = test_model(nn, user_input)
        print(f"Predicted Brand: {predicted_brand}")

# Neural Network parameters
input_size = len(vocabulary)
hidden_size = 10
output_size = len(car_brands)

# Initialize the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Train the neural network
nn.train(X, y, epochs=1000, learning_rate=0.1)

# Run the prediction loop
if __name__ == "__main__":
    predict_from_input()
