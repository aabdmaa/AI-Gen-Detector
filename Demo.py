import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    df = pd.read_csv('data_collection_cleaned.csv')
    return df

def prepare_features(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['Code Sample']).toarray()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Difficulty'])
    return X, y, label_encoder, vectorizer

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_difficulty_with_confidence(code_sample, model, vectorizer, label_encoder):
    ai_keywords = ["tensorflow", "keras", "Conv2D", "MaxPooling2D", "Dense", "Sequential", "fit", "evaluate", 
                   "dlib", "opencv", "torch", "torch.nn", "cv::", "cv::Mat", "TensorFlow"]
    if any(keyword in code_sample.lower() for keyword in ai_keywords):
        return "AI", 100.0

    cpp_keywords_beginner = ["cout", "cin", "main", "#include"]
    if any(keyword in code_sample.lower() for keyword in cpp_keywords_beginner):
        return "Beginner", 100.0
    
    X_new = vectorizer.transform([code_sample]).toarray()
    probas = model.predict_proba(X_new)
    predicted_class = np.argmax(probas)
    confidence = probas[0][predicted_class] * 100
    difficulty = label_encoder.inverse_transform([predicted_class])[0]
    
    return difficulty, confidence

def generate_explanation(difficulty):
    explanations = {
        "Beginner": "This code involves basic syntax like print statements, simple assignments, or basic data types, typical for a beginner.",
        "Intermediate": "This code involves more complex structures like conditionals, loops, or functions, typical for an intermediate level programmer.",
        "Professional": "This code demonstrates advanced concepts such as algorithms, data structures, or optimizations, common among professional programmers.",
        "AI": "This code shows advanced AI/ML constructs such as TensorFlow/Keras layers, neural network training, and data processing typical of machine learning models."
    }
    return explanations.get(difficulty, "No explanation available")

def detect_ai_generated_code(code_sample):
    ai_keywords = [
        "import tensorflow", "import keras", "model.fit", "Conv2D", "Dense", "MaxPooling2D", "Sequential", "evaluate", 
        "tensorflow", "torch", "torch.nn", "dlib", "opencv", "tf.keras", "fit", "keras.models", "cv::", "cv::Mat", "TensorFlow",
        "RandomForestClassifier", "svm", "xgboost", "neural network", "deep learning"
    ]
    ai_likelihood = sum([code_sample.lower().count(keyword) > 0 for keyword in ai_keywords])
    
    if ai_likelihood > 0:
        return 100.0
    else:
        return 0.0

def read_code_from_file(file_name):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, file_name)
    
    try:
        with open(file_path, 'r') as file:
            code = file.read()
        return code
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' could not be found. Please provide a valid file name.")
        return None
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

def detect_language(code_sample):
    code_sample = code_sample.strip().lower()
    if 'import' in code_sample or 'def' in code_sample or 'print(' in code_sample:
        return 'Python'
    if 'public class' in code_sample and 'system.out.println' in code_sample:
        return 'Java'
    if 'class' in code_sample and 'static void main' in code_sample:
        return 'Java'
    if '#include' in code_sample or 'int main' in code_sample:
        return 'C++'
    return 'Unknown'

def estimate_code_difficulty(code_sample):
    beginner_keywords = ['print', 'for', 'while', 'def', 'int', 'char', 'str', 'cout', 'cin', '#include', 'input']
    intermediate_keywords = ['class', 'self', 'try', 'except', 'import', 'def', 'list', 'dict', 'lambda', 'filter', 'map', 'assert', 'collections', 'defaultdict', 'range', 'set']
    professional_keywords = [
        'dynamic programming', 'graph', 'backtracking', 'heap', 'graph theory', 'binary search', 'threading', 'multiprocessing', 
        'asyncio', 'recursion', 'neural network', 'deep learning', 'machine learning', 'optimization', 'concurrency', 'dp', 'memoization',
        'advanced data structures', 'b-tree', 'red-black tree', 'avl tree', 'dijkstra', 'a*', 'greedy algorithm', 'bellman-ford', 'floyd-warshall',
        'big o notation', 'space complexity', 'time complexity', 'design patterns', 'singleton', 'factory', 'observer', 'visitor', 'strategy'
    ]
    cpp_professional_keywords = ['unordered_map', 'priority_queue', 'graph', 'dijkstra', 'floyd-warshall', 'a*', 'backtracking', 'heap', 'binary search', 'advanced data structures', 'graph traversal', 'min-heap', 'max-heap']

    if any(keyword in code_sample.lower() for keyword in beginner_keywords):
        return "Beginner"
    elif any(keyword in code_sample.lower() for keyword in intermediate_keywords):
        return "Intermediate"
    elif any(keyword in code_sample.lower() for keyword in professional_keywords) or any(keyword in code_sample.lower() for keyword in cpp_professional_keywords):
        return "Professional"
    return "AI"

def main():
    print("--------------------------------------------------")
    print("Welcome to the AI Gen Classifier!")

    file_name = input("Please provide the code file name (e.g., code_sample.py): ")
    
    try:
        print("\nProcessing your code sample...\n")
        
        df = load_data()
        X, y, label_encoder, vectorizer = prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        user_input = read_code_from_file(file_name)

        if user_input:
            difficulty, confidence = predict_difficulty_with_confidence(user_input, model, vectorizer, label_encoder)
            explanation = generate_explanation(difficulty)
            ai_likelihood = detect_ai_generated_code(user_input)
            language = detect_language(user_input)

            print("--------------------------------------------------")
            print(f"Code File: {file_name}")
            print(f"Detected Programming Language: {language}")
            print("--------------------------------------------------")
            print(f"Predicted Difficulty: {difficulty}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"\nExplanation: {explanation}")

            print("\nAI Generation Detection:")
            print(f"Likelihood of AI Generation: {ai_likelihood:.2f}%")
            if ai_likelihood == 100:
                print("Explanation: This code shows advanced AI/ML patterns typical of machine learning models.")

            print("--------------------------------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
    