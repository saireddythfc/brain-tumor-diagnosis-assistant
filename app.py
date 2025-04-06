import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import base64
import os
import google.generativeai as genai
from custom_cnn import CustomCNN
import timm
import matplotlib.pyplot as plt

# ---------- App Configuration ----------
st.set_page_config(page_title="Brain MRI Diagnosis Assistant", layout="wide")
st.title("🧠 Brain MRI Diagnosis Assistant")


# ---------- Load Model ----------
@st.cache_resource
def load_model(name):
    if name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 4)
        model.load_state_dict(torch.load("resnet50_finetuned.pth", map_location="cpu"))
        target_layer = model.layer4[-1]
    elif name == "CustomCNN":
        model = CustomCNN(num_classes=4)
        model.load_state_dict(torch.load("customcnn_finetuned.pth", map_location="cpu"))
        target_layer = model.features[-1]
    elif name == "Xception":
        model = timm.create_model("xception", pretrained=False, num_classes=4)
        target_layer = None  # Grad-CAM not supported for now
        model.load_state_dict(torch.load("xception_finetuned.pth", map_location="cpu"))
    model.eval()
    return model, target_layer


# --- Saliency Map ---
def get_saliency_map(model, image_tensor, class_idx=None):
    image_tensor = image_tensor.unsqueeze(0).requires_grad_()
    output = model(image_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    score = output[0, class_idx]
    score.backward()
    saliency = image_tensor.grad.data.abs()
    saliency, _ = torch.max(saliency, dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency, class_idx


# --- Grad-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(
            self.target_layer.register_backward_hook(backward_hook)
        )

    def generate(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.unsqueeze(0).requires_grad_()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()


class_names = ["glioma", "meningioma", "no tumor", "pituitary"]
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

# ---------- Model & LLM Selection ----------
model_name = st.sidebar.selectbox(
    "🧬 Select Model", ["ResNet50", "Xception", "CustomCNN"]
)
model, target_layer = load_model(model_name)

llm_option = st.sidebar.selectbox(
    "🧠 Select Multimodal LLM",
    ["Gemini 1.5 Flash", "GPT-4 Vision (coming soon)", "LLaVA (coming soon)"],
)

# ---------- Upload MRI Image ----------
uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI Image", type=["jpg", "jpeg", "png"]
)
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Scan", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    pred_label = class_names[predicted.item()]
    st.success(f"✅ **Prediction:** {pred_label}")

    # ---------- Visual Explanation ----------
    vis_option = st.radio("🧪 Visualization Type:", ["Saliency Map", "Grad-CAM"])
    cam_image = None

    if vis_option == "Grad-CAM" and target_layer:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=False)
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        rgb_image = np.array(image.resize((256, 256))) / 255.0
        cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        st.image(cam_image, caption="🔍 Grad-CAM", use_column_width=True)

    elif vis_option == "Saliency Map":
        heatmap, class_idx = get_saliency_map(model, transform(image))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(heatmap, cmap="hot")
        ax.set_title(f"{vis_option} - {class_names[class_idx]}")
        ax.axis("off")
        st.pyplot(fig)
        # st.image(saliency_img, caption="🧠 Saliency Map", use_column_width=True)

    if st.toggle("💾 Save Visualization"):
        save_path = f"saved_visual_{model_name.lower()}.png"
        if cam_image is not None:
            Image.fromarray(cam_image).save(save_path)
        else:
            Image.fromarray((saliency_img * 255).astype(np.uint8)).save(save_path)
        st.success(f"Saved to `{save_path}`")

    # ---------- Google API Key ----------
    st.subheader("🔐 Google API Key Required")
    api_key = st.text_input("Enter your Google API key:", type="password")
    enable_llm = api_key.strip() != ""

    if enable_llm:
        genai.configure(api_key=api_key)

        # ---------- Generate Report ----------
        if st.button("📝 Generate Report with Gemini"):
            st.info("Generating medical report...")
            bytes_data = uploaded_file.getvalue()

            gemini_image = {"mime_type": uploaded_file.type, "data": bytes_data}

            prompt = f"""
You are a medical assistant specialized in analyzing brain MRI scans.
Given this scan and the model's prediction: **{pred_label}**, write a comprehensive report including:

1. Model's Prediction
2. Additional Insights based on the scan
3. Historical case references or trends
4. Suggested next steps for the patient
5. Recommendations for the doctor

Format the response with headings and be medically accurate.
"""

            try:
                model_llm = genai.GenerativeModel("gemini-1.5-flash")
                response = model_llm.generate_content([prompt, gemini_image])
                report = response.text
                st.session_state.report_text = report
                st.subheader("📄 Medical Report")
                st.markdown(report)
            except Exception as e:
                st.error(f"Failed to generate report: {e}")

        # ---------- Interactive Chat ----------
        st.subheader("💬 Chat with Gemini about the Report & Image")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Your question:", key="chat_input")

        # Suggested question buttons
        st.markdown("#### 💡 Suggested Questions:")
        questions = [
            "What does the impression suggest?",
            "What are common outcomes for this diagnosis?",
            "What further tests are usually done in such cases?",
            "Can you summarize the report in simple terms?",
        ]
        for q in questions:
            if st.button(q, key=q):
                if st.button(q):
                    if "chat_input" not in st.session_state:
                        st.session_state["chat_input"] = q
                    else:
                        st.session_state["chat_input_preload"] = (
                            q  # Store preload separately
                        )

        if st.button("Send"):
            if user_input:

                report_text = st.session_state.get("report_text", "")
                if report_text:
                    report_context = f"""Here is the prior report:

                {report_text}"""
                else:
                    report_context = ""

                st.session_state.chat_history.append(("user", user_input))
                full_prompt = f"""
                                You are an assistant analyzing a brain MRI scan.
                                Here is the model's prediction: **{pred_label}**.
                                {report_context}
                                Now answer the following question from the user:
                              """
                try:
                    model_llm = genai.GenerativeModel("gemini-1.5-flash")
                    chat_session = model_llm.start_chat(history=[])
                    response = chat_session.send_message([full_prompt, user_input])
                    answer = response.text
                    st.session_state.chat_history.append(("bot", answer))
                except Exception as e:
                    st.session_state.chat_history.append(("bot", f"Error: {e}"))

        # Display chat history
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Gemini:** {msg}")
    else:
        st.warning(
            "🔐 Please enter your Google API key to use Gemini-powered features."
        )
