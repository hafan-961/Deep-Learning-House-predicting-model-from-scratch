# 🧠 Deep Learning House Price Prediction (From Scratch)

A complete end-to-end project where a Deep Learning model is **built, trained, and deployed from scratch** — without using high-level frameworks like TensorFlow or PyTorch.  
The model predicts **house prices** based on given input features and is **deployed live** on the web using **JavaScript** for real-time predictions.

---

## 🚀 Project Overview

This project demonstrates the **entire pipeline** of a Deep  learning workflow:

1. **Model Building (from scratch)**  
   - Implemented all neural network operations manually (forward pass, backpropagation, weight updates).
   - 2 hidden layers with 32 neuron's on first layer and 16 neuron's on second layer.
   - Trained on a dataset of house prices using gradient descent optimization.  

2. **Parameter Saving**  
   - After training, all learned **weights** and **biases** are saved to a JSON file (`model_parameters.json`) for reuse.  

3. **Web Deployment**  
   - A **JavaScript-based frontend** loads the saved parameters.  
   - The network’s prediction logic is reimplemented in JS to make **live predictions** directly in the browser.  

---

## 🏗️ Technologies Used

### 💻 Model Training
- Python  
- NumPy
- Pandas
- sklearn
- Matplotlib (for visualization)
- Google Colab

### 🌐 Deployment
- HTML, CSS, JavaScript  
- JSON (for saving weights and biases)

---

## 📂 Project Structure

📦 Deep-Learning-House-Price-Predictor
├── 📓 house_prediction(deepLearning from scratch).ipynb # Google Colab for model training
├── 📄 model_parameters.json # Saved weights and biases
├── 💻 index.html # Main web page for predictions
├── 🧠 script.js # JavaScript logic for loading model and predicting
├── 🎨 style.css # CSS file for webpage styling
└── 🪶 README.md # Documentation file

