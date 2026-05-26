import streamlit as st
import numpy as np
import torch
import os
from PIL import Image
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
from model import MNISTCNN

def preprocess_digit(img_np):
    """
    Preprocess a numpy image array (grayscale or RGBA) to match MNIST formatting:
    - Finds the bounding box of the digit and crops it.
    - Resizes the digit to fit a 20x20 box preserving aspect ratio.
    - Computes the center of mass of the resized digit.
    - Places the 20x20 digit on a blank 28x28 canvas centered on its center of mass.
    - Normalizes the pixels to match MNIST statistics.
    """
    # 1. Grayscale conversion if input is RGBA
    if len(img_np.shape) == 3:
        if img_np.shape[2] == 4:
            # Convert RGBA to grayscale: take maximum of RGB channels to get stroke intensity
            img = np.max(img_np[:, :, :3], axis=2).astype(np.uint8)
        else:
            img = np.max(img_np, axis=2).astype(np.uint8)
    else:
        img = img_np.astype(np.uint8)
        
    # 2. Find bounding box of the digit
    rows = np.any(img > 10, axis=1) # threshold low noise
    cols = np.any(img > 10, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None, None
        
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    digit = img[ymin:ymax+1, xmin:xmax+1]
    
    # 3. Resize to fit inside a 20x20 box preserving aspect ratio
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = int(round(20.0 * w / h))
        new_w = max(1, new_w)
    else:
        new_w = 20
        new_h = int(round(20.0 * h / w))
        new_h = max(1, new_h)
        
    digit_pil = Image.fromarray(digit)
    digit_resized = digit_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
    digit_arr = np.array(digit_resized)
    
    # 4. Center the digit using center of mass on a blank 28x28 canvas
    new_img = np.zeros((28, 28), dtype=np.float32)
    
    # Calculate center of mass of the resized digit
    y_indices, x_indices = np.where(digit_arr > 10)
    if len(y_indices) > 0:
        weights = digit_arr[y_indices, x_indices]
        cy = np.average(y_indices, weights=weights)
        cx = np.average(x_indices, weights=weights)
    else:
        cy, cx = new_h / 2.0, new_w / 2.0
        
    # Calculate placement offset (Target center is 14.0, 14.0)
    start_y = int(round(14.0 - cy))
    start_x = int(round(14.0 - cx))
    
    # Copy digit onto blank canvas
    for y in range(new_h):
        for x in range(new_w):
            target_y = start_y + y
            target_x = start_x + x
            if 0 <= target_y < 28 and 0 <= target_x < 28:
                new_img[target_y, target_x] = digit_arr[y, x]
                
    # Normalize to 0-1
    new_img = new_img / 255.0
    
    # Apply standard MNIST normalization constants
    img_normalized = (new_img - 0.1307) / 0.3081
    tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0) # (1, 1, 28, 28)
    
    # Create scaled back image for display preview
    preview_img = Image.fromarray((new_img * 255).astype(np.uint8))
    
    return tensor, preview_img

# 1. Page Configuration & Aesthetic Styles
st.set_page_config(
    page_title="MNIST Interactive Digit Recognizer",
    page_icon="✍️",
    layout="wide"
)

# Inject clean styles compatible with default Streamlit themes
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Apply font to app */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Title styling - clean and bold */
.glow-title {
    font-weight: 800;
    margin-bottom: 5px;
}

.glow-subtitle {
    margin-bottom: 25px;
    opacity: 0.8;
}

/* Premium prediction metric container */
.prediction-card {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.06) 0%, rgba(147, 51, 234, 0.06) 100%);
    border: 1px solid rgba(79, 70, 229, 0.18);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    margin-bottom: 15px;
    transition: all 0.3s ease;
}

.prediction-card:hover {
    border-color: rgba(147, 51, 234, 0.3);
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.08);
    transform: translateY(-2px);
}

.prediction-digit-large {
    font-size: 80px;
    font-weight: 800;
    background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 5px 0;
    line-height: 1.1;
}

.prediction-label {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.7;
}

.prediction-confidence-large {
    font-size: 16px;
    font-weight: 500;
    opacity: 0.9;
}

/* Fine-grained image hover effects */
.stImage img {
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    border-radius: 8px;
    border: 1px solid rgba(128, 128, 128, 0.15);
}

.stImage img:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
}
</style>
""",unsafe_allow_html=True)

# 2. Load Model Helper
@st.cache_resource
def load_trained_model(mtime):
    model = MNISTCNN()
    if os.path.exists("mnist_cnn.pth"):
        model.load_state_dict(torch.load("mnist_cnn.pth", map_location=torch.device('cpu')))
        model.eval()
        return model, True
    return model, False

mtime = os.path.getmtime("mnist_cnn.pth") if os.path.exists("mnist_cnn.pth") else 0
model, model_loaded = load_trained_model(mtime)

# 3. Sidebar Panel
st.sidebar.markdown("<h2 class='glow-title' style='font-size: 24px;'>🤖 Model Settings</h2>", unsafe_allow_html=True)

if model_loaded:
    st.sidebar.success("Model status: LOADED")
else:
    st.sidebar.warning("Model status: NOT FOUND\nPlease train the model first.")
    
st.sidebar.markdown("---")
st.sidebar.write("### Canvas Parameters")
stroke_width = st.sidebar.slider("Brush Thickness", min_value=10, max_value=30, value=18, step=1)
drawing_mode = st.sidebar.selectbox("Drawing Tool", ("freedraw", "transform"))

st.sidebar.markdown("---")
with st.sidebar.expander("🔍 Model Architecture Details", expanded=False):
    st.info(
        "**Architecture: CNN**\n\n"
        "- **Conv2D (1 → 32)** | 3x3 kernel\n"
        "- **ReLU & MaxPool2d** | 2x2 stride\n"
        "- **Conv2D (32 → 64)** | 3x3 kernel\n"
        "- **ReLU & MaxPool2d** | 2x2 stride\n"
        "- **Dropout (25%)**\n"
        "- **Dense Linear (3136 → 128)** | ReLU\n"
        "- **Dropout (50%)**\n"
        "- **Dense Linear (128 → 10)**\n"
    )

# Check if training plot exists to display in sidebar
if os.path.exists("training_history.png"):
    st.sidebar.markdown("---")
    with st.sidebar.expander("📈 Training Performance", expanded=False):
        st.image("training_history.png", width='stretch')

# 4. Main Panel Header
st.markdown("<h1 class='glow-title'>✍️ Handwritten Digit Recognition System</h1>", unsafe_allow_html=True)
st.markdown("<p class='glow-subtitle'>Draw a digit on the canvas or upload an image to watch the CNN model classify it and visualize its inner features.</p>", unsafe_allow_html=True)

# Layout division
col_canvas, col_results = st.columns([1.1, 1], gap="large")

# Holds final input grayscale image
input_tensor = None
raw_img_for_view = None

with col_canvas:
    with st.container(border=True):
        st.markdown("### 🎨 Input Canvas")
        
        tab_draw, tab_upload = st.tabs(["🖌️ Draw Digit", "📤 Upload Image"])
        
        with tab_draw:
            draw_col, preview_col = st.columns([1.6, 1])
            
            with draw_col:
                # Create canvas for drawing
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=stroke_width,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode=drawing_mode,
                    update_streamlit=True,
                    key="mnist_canvas",
                )
            
            draw_tensor = None
            draw_preview = None
            if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 10):
                draw_tensor, draw_preview = preprocess_digit(canvas_result.image_data)
                input_tensor = draw_tensor
                raw_img_for_view = draw_preview
                
            with preview_col:
                st.markdown("<div style='text-align: center; display: flex; flex-direction: column; align-items: center;'>", unsafe_allow_html=True)
                st.markdown("##### 🕵️ What the CNN sees")
                if draw_preview is not None:
                    st.image(draw_preview, caption="Downsampled 28x28 Input", width='stretch')
                else:
                    placeholder = Image.new("L", (28, 28), color=0)
                    st.image(placeholder, caption="Draw a digit to preview", width='stretch')
                st.markdown("</div>", unsafe_allow_html=True)
                
        with tab_upload:
            upload_col, preview_col = st.columns([1.6, 1])
            
            with upload_col:
                uploaded_file = st.file_uploader("Upload an image of a digit (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
                
                upload_tensor = None
                upload_preview = None
                uploaded_img = None
                if uploaded_file is not None:
                    uploaded_img = Image.open(uploaded_file)
                    
                    # Convert to grayscale numpy array
                    img_gray = uploaded_img.convert('L')
                    img_arr = np.array(img_gray)
                    
                    # Smart Color Inversion:
                    # MNIST is white digits on black background. If the uploaded image has a light background, invert it.
                    border_pixels = np.concatenate([
                        img_arr[0, :], img_arr[-1, :], img_arr[:, 0], img_arr[:, -1]
                    ])
                    if np.mean(border_pixels) > 127:
                        img_arr = 255 - img_arr
                    
                    # Preprocess digit using recentering & scaling
                    upload_tensor, upload_preview = preprocess_digit(img_arr)
                    input_tensor = upload_tensor
                    raw_img_for_view = upload_preview
                    
            with preview_col:
                st.markdown("<div style='text-align: center; display: flex; flex-direction: column; align-items: center;'>", unsafe_allow_html=True)
                st.markdown("##### 🕵️ What the CNN sees")
                if upload_preview is not None:
                    st.image(upload_preview, caption="Downsampled 28x28 Input", width='stretch')
                    if uploaded_img is not None:
                        st.image(uploaded_img, caption="Original Image", width='stretch')
                else:
                    placeholder = Image.new("L", (28, 28), color=0)
                    st.image(placeholder, caption="Upload an image to preview", width='stretch')
                st.markdown("</div>", unsafe_allow_html=True)

with col_results:
    with st.container(border=True):
        st.markdown("### 📊 Model Analysis")
        
        if not model_loaded:
            st.warning("⚠️ The PyTorch CNN model is currently training or not yet compiled. Run the training script first.")
        elif input_tensor is not None:
            # Run model inference
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.softmax(logits, dim=1).squeeze().tolist()
                predicted_digit = np.argmax(probs)
                confidence = probs[predicted_digit] * 100
            
            # Display Prediction Card
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-label">Predicted Classification</div>
                <div class="prediction-digit-large">{predicted_digit}</div>
                <div class="prediction-confidence-large">Confidence: <b>{confidence:.2f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display Plotly Probabilities
            st.markdown("<h5>Classification Probabilities</h5>", unsafe_allow_html=True)
            
            # Color palette matching the premium gradient theme
            # Highlight the predicted digit with indigo/purple and make others light indigo
            colors = ['rgba(79, 70, 229, 0.25)'] * 10
            colors[predicted_digit] = 'rgba(147, 51, 234, 1.0)'
            
            fig = go.Figure(go.Bar(
                x=[str(i) for i in range(10)],
                y=probs,
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(0,0,0,0)', width=0)
                ),
                text=[f"{p*100:.0f}%" if p > 0.04 else "" for p in probs],
                textposition='outside'
            ))
            
            fig.update_layout(
                xaxis=dict(
                    title='Digit Class',
                    showgrid=False,
                    tickmode='linear'
                ),
                yaxis=dict(
                    title='Probability',
                    range=[0, 1.1],
                    gridcolor='rgba(128, 128, 128, 0.1)',
                    tickformat='.0%'
                ),
                margin=dict(l=40, r=10, t=25, b=40),
                height=240,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, width='stretch', theme="streamlit")
            
        else:
            st.info("✍️ Draw a digit or upload an image to see prediction analysis.")

# 5. Visualizing Neural Network Feature Maps (Intermediate Activation Visualizer)
if model_loaded and input_tensor is not None:
    with st.container(border=True):
        st.markdown("### 🔍 Convolutional Layers - Inner Activations")
        st.markdown("<p style='opacity: 0.8; margin-bottom: 20px;'>These feature maps illustrate what parts of your drawing trigger specific filters in the neural network. Conv1 extracts basic edge features; Conv2 aggregates those into complex shapes.</p>", unsafe_allow_html=True)
        
        act1, act2 = model.get_activations(input_tensor)
        
        # Tab layout to separate Conv layers
        tab_conv1, tab_conv2 = st.tabs(["📷 Conv Layer 1 Features (Filters 1-16)", "📸 Conv Layer 2 Features (Filters 1-16)"])
        
        with tab_conv1:
            st.write("First Convolutional Layer (Activations shape: 32 channels, showing first 16)")
            cols = st.columns(8)
            for idx in range(16):
                col = cols[idx % 8]
                feat = act1[0, idx].detach().numpy()
                
                # Normalize to 0-255 for clean visual rendering
                f_min, f_max = feat.min(), feat.max()
                if f_max > f_min:
                    feat_scaled = (feat - f_min) / (f_max - f_min)
                else:
                    feat_scaled = np.zeros_like(feat)
                    
                feat_img = Image.fromarray((feat_scaled * 255).astype(np.uint8))
                # Upscale for better viewing resolution
                feat_img = feat_img.resize((112, 112), Image.Resampling.NEAREST)
                col.image(feat_img, caption=f"Filter {idx+1}", width='stretch')
                
        with tab_conv2:
            st.write("Second Convolutional Layer (Activations shape: 64 channels, showing first 16)")
            cols = st.columns(8)
            for idx in range(16):
                col = cols[idx % 8]
                feat = act2[0, idx].detach().numpy()
                
                # Normalize to 0-255 for clean visual rendering
                f_min, f_max = feat.min(), feat.max()
                if f_max > f_min:
                    feat_scaled = (feat - f_min) / (f_max - f_min)
                else:
                    feat_scaled = np.zeros_like(feat)
                    
                feat_img = Image.fromarray((feat_scaled * 255).astype(np.uint8))
                # Upscale for better viewing resolution
                feat_img = feat_img.resize((112, 112), Image.Resampling.NEAREST)
                col.image(feat_img, caption=f"Filter {idx+1}", width='stretch')
