# app.py
import os
import sys
import time
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- Set Up Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. CORE ARCHITECTURES (Embedded directly)
# ==========================================

class ActivationFactory:
    @staticmethod
    def get(name):
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {name}")

class CustomMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[512, 256], num_classes=10, activation="relu", dropout=0.0):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(ActivationFactory.get(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

class CustomCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, activation="relu", dropout=0.0):
        super().__init__()
        act = activation
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            ActivationFactory.get(act),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            ActivationFactory.get(act),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            ActivationFactory.get(act),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ==========================================
# 2. STREAMLIT USER INTERFACE DEPLOYMENT
# ==========================================

st.set_page_config(page_title="NeuralForge: Architecture Benchmark Lab", layout="wide")

st.title("🛡️ NeuralForge: CNN & MLP Architecture Benchmark Lab")
st.write("A deep learning testing suite evaluating custom architectures, activation profiles, and inference latencies from scratch.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Model Configuration")
dataset_choice = st.sidebar.selectbox("Target Dataset", ["MNIST (Grayscale)", "CIFAR-10 (RGB)"])
arch_choice = st.sidebar.selectbox("Architecture Type", ["CNN", "MLP"])
activation_choice = st.sidebar.selectbox("Activation Function Used", ["ReLU", "GELU", "Swish"])
dropout_rate = st.sidebar.slider("Dropout Regularization", 0.0, 0.5, 0.3, step=0.1)

# Target check-point configurations matching training specs
run_name = f"{dataset_choice.split(' ')[0].lower()}_{arch_choice}_{activation_choice.lower()}_dr{dropout_rate}"
checkpoint_path = os.path.join(CURRENT_DIR, "checkpoints", f"{run_name}_best.pt")

st.sidebar.info(f"**Target Checkpoint Path:**\n`{checkpoint_path}`")

# --- MODEL LOADING LAYER ---
@st.cache_resource
def load_benchmark_model(dataset, arch, activation, dropout, path):
    in_channels = 1 if "MNIST" in dataset else 3
    input_dim = 784 if "MNIST" in dataset else 3072
    num_classes = 10
    
    if arch == "MLP":
        model = CustomMLP(input_dim=input_dim, activation=activation, dropout=dropout)
    else:
        model = CustomCNN(in_channels=in_channels, activation=activation, dropout=dropout)
        
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            status = "✨ Loaded trained weight parameters cleanly."
        except Exception as e:
            status = f"⚠️ Weight load failed: {str(e)}"
    else:
        status = "⚠️ Running with randomly initialized structural weights (No checkpoint found)."
        
    model.to(DEVICE)
    model.eval()
    return model, status

model, weights_status = load_benchmark_model(dataset_choice, arch_choice, activation_choice, dropout_rate, checkpoint_path)
st.sidebar.caption(f"Status: {weights_status}")

# --- UI SPLIT LAYOUT LAYER ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Image Upload & Processing")
    uploaded_file = st.file_uploader("Upload a sample image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Input Target", width=240)
        
        if "MNIST" in dataset_choice:
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            classes = [str(i) for i in range(10)]
        else:
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # --- LIVE INFERENCE RUN ---
        with torch.no_grad():
            start_time = time.perf_counter()
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            end_time = time.perf_counter()
            
        latency_ms = (end_time - start_time) * 1000
        prediction_idx = np.argmax(probabilities)
        predicted_class = classes[prediction_idx]

with col2:
    st.subheader("📊 Real-time Inference Analytics")
    if uploaded_file is not None:
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Predicted Label", predicted_class.upper())
        m_col2.metric("Inference Latency", f"{latency_ms:.2f} ms")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        y_pos = np.arange(len(classes))
        ax.barh(y_pos, probabilities, color='#4A90E2', edgecolor='none')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes)
        ax.invert_yaxis()
        ax.set_xlabel('Confidence Probability Score')
        ax.set_xlim(0.0, 1.0)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Please upload an image file to trigger the model profiling layer.")