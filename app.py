import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import joblib
import os
import gdown

# PAGE CONFIG
st.set_page_config(
    page_title="CIFAR100 + Credit Card Fraud & Segmentation",
    layout="centered"
)

# SIDEBAR
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", [
    "CIFAR-100 Image Classifier",
    "Credit Card Fraud Detection",
    "Customer Segmentation"
])

st.sidebar.info("This app combines Computer Vision and Machine Learning models into one interface.")

# MAIN TITLE
st.title("AI Multi-Model App")
st.write("A simple but powerful web app combining **image classification**, **fraud detection**, and **customer segmentation**.")


# LOAD COMPUTER VISION MODEL

def load_cv_model():
    MODEL_PATH = "resnet50_cifar100.pth"
    URL = "https://drive.google.com/uc?id=1UwmbwsiKCFrAmmX3rppB4Nr0Y8vompwy"

    # Download model if missing
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model... please wait.")
        gdown.download(URL, MODEL_PATH, quiet=False)

    # Load model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

cv_model = load_cv_model()

cv_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

LABELS = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
    "chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard",
    "lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain",
    "mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree",
    "plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket",
    "rose","sea","seal","shark","shrew","skyscraper","snail","snake","spider","squirrel",
    "streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor","train",
    "trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm"
]


# LOAD CREDIT CARD MODELS

def load_cc_models():
    model_path = "Credit_Fraud.joblib"
    if not os.path.exists(model_path):
        st.error("Credit card model NOT found. Put Credit_Fraud.joblib in this folder.")
        return None
    return joblib.load(model_path)

cc_models = load_cc_models()


# CIFAR-100 IMAGE CLASSIFIER

if section == "CIFAR-100 Image Classifier":
    st.header("CIFAR-100 Image Classifier")
    st.subheader("Classify objects from 100 categories")
    st.write("Upload an image and let the model identify the object. This ResNet-50 model was fine-tuned on the CIFAR-100 dataset.")

    uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", width=250)

        if st.button("Classify Image"):
            if cv_model is not None:
                img_t = cv_transform(image).unsqueeze(0)

                with torch.no_grad():
                    output = cv_model(img_t)
                    probs = torch.softmax(output, dim=1).numpy()[0]

                pred_index = np.argmax(probs)
                pred_label = LABELS[pred_index]
                confidence = probs[pred_index] * 100

                st.success(f"This is a **{pred_label}** ({confidence:.2f}% confidence)")


# CREDIT CARD FRAUD DETECTION

elif section == "Credit Card Fraud Detection":
    st.header("Credit Card Fraud Detection")
    st.subheader("Detect fraudulent transactions instantly")
    st.write("Enter transaction values below. This model predicts whether the transaction is **fraudulent or legitimate**.")

    if cc_models is not None:
        col1, col2 = st.columns(2)
        amount = col1.number_input("Amount", value=100.0)
        time_val = col2.number_input("Time", value=5000.0)

        v1 = st.number_input("V1", value=0.0)
        v2 = st.number_input("V2", value=0.0)
        v3 = st.number_input("V3", value=0.0)
        v4 = st.number_input("V4", value=0.0)

        if st.button("Predict Fraud"):
            X = np.array([[time_val, amount, v1, v2, v3, v4]])
            pred = cc_models['fraud'].predict(X)

            if pred[0] == 1:
                st.error("Fraud Detected!")
            else:
                st.success("Legit Transaction")


# CUSTOMER SEGMENTATION

elif section == "Customer Segmentation":
    st.header("Customer Segmentation")
    st.subheader("Group customers based on behavior")
    st.write("This model classifies customers into behavioral segments to support better marketing decisions.")

    if cc_models is not None:
        col1, col2 = st.columns(2)
        amount_s = col1.number_input("Amount", value=100.0, key="seg_amount")
        time_s = col2.number_input("Time", value=5000.0, key="seg_time")

        v1_s = st.number_input("V1", value=0.0, key="seg_v1")
        v2_s = st.number_input("V2", value=0.0, key="seg_v2")
        v3_s = st.number_input("V3", value=0.0, key="seg_v3")
        v4_s = st.number_input("V4", value=0.0, key="seg_v4")

        if st.button("🔹 Segment Customer"):
            X_seg = np.array([[time_s, amount_s, v1_s, v2_s, v3_s, v4_s]])
            segment = cc_models['segmentation'].predict(X_seg)

            st.info(f"Customer belongs to **Segment {segment[0]}**")


# FOOTER

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:13px;'>
        Built by Aishat Sunday
    </div>
    """,
    unsafe_allow_html=True
)
