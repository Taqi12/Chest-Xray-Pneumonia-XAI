# app.py - Chest X-Ray Pneumonia Detection & Explainable AI (Streamlit App)
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Medical AI - Pneumonia Detection", page_icon="🫁", layout="centered")

st.title("🫁 Chest X-Ray Pneumonia Diagnostic AI")
st.write("Upload a chest X-ray image to get instant AI prediction along with Grad-CAM Explainable AI visualization.")

# 1. Load Trained ResNet50 Model (Cached for performance)
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    # Load your trained weights
    model.load_state_dict(torch.load('resnet50_pneumonia_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

with st.spinner("Loading AI Model... Please wait."):
    model = load_model()

class_names = ['NORMAL', 'PNEUMONIA']

# 2. File Uploader
uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display uploaded image
    st.image(image, caption='Uploaded Chest X-ray', use_column_width=True)
    
    if st.button("Run AI Diagnosis & Grad-CAM"):
        with st.spinner("Analyzing X-ray and generating Explainable AI heatmap..."):
            
            # Preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0)
            
            # Grad-CAM Hooks Setup
            activation = None
            gradient = None

            def forward_hook(module, input, output):
                global activation
                activation = output.detach()

            def backward_hook(module, grad_in, grad_out):
                global gradient
                gradient = grad_out[0].detach()

            target_layer = model.layer4[-1]
            h1 = target_layer.register_forward_hook(forward_hook)
            h2 = target_layer.register_full_backward_hook(backward_hook)

            # Prediction & Backward pass
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)
            pred_label = class_names[pred_idx.item()]
            
            model.zero_grad()
            output[0, pred_idx.item()].backward()

            h1.remove()
            h2.remove()

            # Generate Heatmap
            pooled_gradients = torch.mean(gradient, dim=[0, 2, 3])
            for i in range(activation.shape[1]):
                activation[0, i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(activation, dim=1).squeeze().cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() != 0:
                heatmap /= heatmap.max()

            heatmap = cv2.resize(heatmap, (224, 224))
            
            # Convert original image for plotting
            img_resized = image.resize((224, 224))
            img_np = np.array(img_resized) / 255.0

            # Display Results
            st.markdown("---")
            if pred_label == "PNEUMONIA":
                st.error(f"### Prediction: **{pred_label}** (Confidence: {confidence.item()*100:.2f}%)")
            else:
                st.success(f"### Prediction: **{pred_label}** (Confidence: {confidence.item()*100:.2f}%)")
            
            # Plotting Heatmap side-by-side using Matplotlib
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            
            axes[0].imshow(img_np)
            axes[0].set_title("Resized X-ray")
            axes[0].axis('off')

            axes[1].imshow(img_np)
            axes[1].imshow(heatmap, cmap='jet', alpha=0.5)
            axes[1].set_title(f"Grad-CAM Focus ({pred_label})")
            axes[1].axis('off')

            st.pyplot(fig)