# ==============================================================================
# app.py
# Skin Disease Detection — Streamlit Web Application
# CNN-ANFIS Hybrid System | Bahria University Islamabad
#
# Deployment notes:
#   - EfficientNetV2M backbone (~480 MB) is downloaded automatically from
#     Keras on first cold start and cached. It is NOT stored in the repo.
#   - The only model files committed to GitHub are:
#       models/anfis_config.pkl        (~1 KB)
#       models/pca_cnn.pkl             (~100 KB)
#       models/scaler_cnn.pkl          (~10 KB)
#       models/cnn_anfis_all_states.pt (~varies, usually < 5 MB)
#   - On Streamlit Community Cloud (1 GB RAM), the first startup may take
#     1-2 minutes while the backbone downloads. Subsequent starts are fast.
#   - Hugging Face Spaces (2 GB RAM) is a more comfortable alternative.
# ==============================================================================

import os
import pickle
import warnings
import numpy as np
from pathlib import Path
from PIL import Image

import streamlit as st

warnings.filterwarnings('ignore')

# ==============================================================================
# Page configuration — must be the first Streamlit call
# ==============================================================================
st.set_page_config(
    page_title="Skin Disease Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Paths
# ==============================================================================
MODELS_DIR = Path("models")

ANFIS_CONFIG_PATH  = MODELS_DIR / "anfis_config.pkl"
ANFIS_STATES_PATH  = MODELS_DIR / "cnn_anfis_all_states.pt"
PCA_PATH           = MODELS_DIR / "pca_cnn.pkl"
SCALER_PATH        = MODELS_DIR / "scaler_cnn.pkl"

# ==============================================================================
# ANFIS architecture — must match the definition used during training
# ==============================================================================
import torch
import torch.nn as nn


class GaussianMembershipLayer(nn.Module):
    """Layer 1 of ANFIS: Gaussian membership functions (one per rule per input)."""

    def __init__(self, n_inputs, n_rules):
        super(GaussianMembershipLayer, self).__init__()
        self.centres    = nn.Parameter(torch.zeros(n_rules, n_inputs))
        self.log_sigmas = nn.Parameter(torch.zeros(n_rules, n_inputs))

    def forward(self, x):
        x_expand = x.unsqueeze(1)
        c_expand = self.centres.unsqueeze(0)
        sigmas   = torch.exp(torch.clamp(self.log_sigmas, -3.0, 3.0)).unsqueeze(0)
        return torch.exp(-0.5 * ((x_expand - c_expand) / (sigmas + 1e-8)) ** 2)


class ANFISBinaryClassifier(nn.Module):
    """Zero-order Takagi-Sugeno ANFIS for binary (one-vs-rest) classification."""

    def __init__(self, n_inputs, n_rules):
        super(ANFISBinaryClassifier, self).__init__()
        self.n_inputs = n_inputs
        self.n_rules  = n_rules
        self.membership_layer = GaussianMembershipLayer(n_inputs, n_rules)
        self.consequents      = nn.Parameter(torch.zeros(n_rules))

    def compute_normalised_firing_strengths(self, x):
        membership  = self.membership_layer(x)
        log_firing  = torch.sum(torch.log(membership + 1e-10), dim=2)
        log_norm    = log_firing - torch.logsumexp(log_firing, dim=1, keepdim=True)
        return torch.exp(log_norm)

    def forward(self, x):
        w_bar  = self.compute_normalised_firing_strengths(x)
        return torch.sum(w_bar * self.consequents, dim=1)

    def get_top_rule_weights(self, n_top=5):
        """Return indices and weights of the n_top most influential rules."""
        weights  = self.consequents.detach().numpy()
        top_idx  = np.argsort(np.abs(weights))[::-1][:n_top]
        return top_idx, weights[top_idx]


# ==============================================================================
# Cached resource loaders
# ==============================================================================

@st.cache_resource(show_spinner="Loading EfficientNetV2M backbone ...")
def load_feature_extractor():
    """
    Load EfficientNetV2M pretrained on ImageNet as a frozen feature extractor.
    The backbone weights are downloaded from Keras servers on first call
    (~480 MB) and cached locally. No backbone weights are stored in this repo.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetV2M

    backbone = EfficientNetV2M(
        include_top=False,
        weights='imagenet',
        input_shape=(480, 480, 3),
        pooling='avg'
    )
    backbone.trainable = False
    return backbone


@st.cache_resource(show_spinner="Loading ANFIS models ...")
def load_anfis_models():
    """
    Load the ANFIS OvR ensemble and supporting preprocessing objects from disk.
    Returns a dict with keys: models, config, pca, scaler.
    """
    if not ANFIS_CONFIG_PATH.exists():
        return None

    with open(str(ANFIS_CONFIG_PATH), 'rb') as f:
        config = pickle.load(f)

    with open(str(PCA_PATH), 'rb') as f:
        pca = pickle.load(f)

    with open(str(SCALER_PATH), 'rb') as f:
        scaler = pickle.load(f)

    all_states = torch.load(str(ANFIS_STATES_PATH), map_location='cpu')

    models = {}
    for class_idx in range(config['num_classes']):
        m = ANFISBinaryClassifier(
            n_inputs=config['n_inputs'],
            n_rules=config['n_rules']
        )
        m.load_state_dict(all_states[class_idx])
        m.eval()
        models[class_idx] = m

    return {'models': models, 'config': config, 'pca': pca, 'scaler': scaler}


# ==============================================================================
# Inference helpers
# ==============================================================================

def preprocess_image_for_inference(pil_image, target_size=480):
    """
    Resize and normalise a PIL image to the format expected by EfficientNetV2M.
    Returns a float32 numpy array of shape [1, H, W, 3] in [0, 1].
    """
    import numpy as np
    rgb_image  = pil_image.convert('RGB')
    resized    = rgb_image.resize((target_size, target_size), Image.LANCZOS)
    img_array  = np.array(resized, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def extract_features_from_image(pil_image, backbone):
    """Run the frozen EfficientNetV2M backbone to get a 1280-dim feature vector."""
    import numpy as np
    import tensorflow as tf

    img_array   = preprocess_image_for_inference(pil_image)
    img_tensor  = tf.constant(img_array)
    features    = backbone(img_tensor, training=False)
    return features.numpy().squeeze()       # [1280]


def run_anfis_inference(raw_features, anfis_bundle):
    """
    Apply the full inference pipeline:
    1. Standardise features with the fitted scaler
    2. Reduce dimensions with the fitted PCA
    3. Run each OvR ANFIS model and collect sigmoid confidence scores
    4. Determine predicted class and flag ambiguous predictions

    Returns: predicted_class_idx, confidence_scores [C], ambiguous (bool)
    """
    scaler  = anfis_bundle['scaler']
    pca     = anfis_bundle['pca']
    models  = anfis_bundle['models']
    n_cls   = len(models)

    scaled  = scaler.transform(raw_features.reshape(1, -1))    # [1, 1280]
    reduced = pca.transform(scaled)                             # [1, n_pca]
    x_tensor = torch.FloatTensor(reduced)                       # [1, n_pca]

    conf_scores = np.zeros(n_cls, dtype=np.float32)

    for class_idx, model in models.items():
        with torch.no_grad():
            raw_out = model(x_tensor)
            conf_scores[class_idx] = torch.sigmoid(raw_out).item()

    predicted_idx = int(np.argmax(conf_scores))

    # Ambiguous if top-2 scores are within 0.10 of each other
    sorted_scores = np.sort(conf_scores)[::-1]
    is_ambiguous  = bool(sorted_scores[0] - sorted_scores[1] < 0.10)

    return predicted_idx, conf_scores, is_ambiguous


# ==============================================================================
# CSS styling
# ==============================================================================

def inject_custom_css():
    """Inject CSS for a clean medical-themed look."""
    st.markdown("""
    <style>
        /* ---- Main background ---- */
        .main { background-color: #f4f6f8; }

        /* ---- Card-style panels ---- */
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 20px 24px;
            margin: 10px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 5px solid #2e7d9e;
        }

        /* ---- Ambiguous warning card ---- */
        .warning-card {
            background: #fff8e1;
            border-radius: 10px;
            padding: 16px 20px;
            margin: 10px 0;
            border-left: 5px solid #f0a500;
        }

        /* ---- Class label headers ---- */
        .prediction-label {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1a5276;
            margin: 0 0 4px 0;
        }

        /* ---- Confidence sub-text ---- */
        .confidence-text {
            font-size: 1.0rem;
            color: #555;
        }

        /* ---- Section headers ---- */
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #2e7d9e;
            border-bottom: 2px solid #d0e8f0;
            padding-bottom: 4px;
            margin-bottom: 12px;
        }

        /* ---- Sidebar styling ---- */
        .css-1d391kg { background-color: #e8f4f8; }

        /* ---- Disclaimer box ---- */
        .disclaimer {
            background: #fdecea;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.85rem;
            color: #7b2a2a;
            border-left: 4px solid #c0392b;
        }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# UI components
# ==============================================================================

def render_confidence_bar_chart(conf_scores, class_names, predicted_idx):
    """
    Display a horizontal bar chart of per-class confidence scores
    using st.progress elements (no extra libraries needed).
    """
    import matplotlib.pyplot as plt

    sorted_order = np.argsort(conf_scores)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ['#1a5276' if i == predicted_idx else '#aed6f1' for i in sorted_order]
    y_pos   = range(len(class_names))

    ax.barh(
        list(y_pos),
        conf_scores[sorted_order],
        color=colors,
        edgecolor='white',
        linewidth=0.8
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([class_names[i] for i in sorted_order], fontsize=11)
    ax.set_xlabel('Confidence Score', fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.set_title('Per-Class Confidence (CNN-ANFIS)', fontsize=12, fontweight='bold')
    ax.axvline(0.5, color='gray', linestyle='--', lw=1, alpha=0.6)
    ax.grid(axis='x', alpha=0.3)

    for idx, score in enumerate(conf_scores[sorted_order]):
        ax.text(score + 0.01, idx, f'{score:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    return fig


def render_membership_function_plot(anfis_model, class_name, pca_features_range):
    """
    Plot the learned Gaussian membership functions for one ANFIS model
    on PCA component 1. Shows the fuzzy partitioning of the feature space.
    """
    import matplotlib.pyplot as plt

    anfis_model.eval()
    x_range = np.linspace(pca_features_range[0], pca_features_range[1], 400)

    centres = anfis_model.membership_layer.centres[:, 0].detach().numpy()
    sigmas  = torch.exp(
        torch.clamp(anfis_model.membership_layer.log_sigmas[:, 0], -3, 3)
    ).detach().numpy()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors  = plt.cm.tab20(np.linspace(0, 1, anfis_model.n_rules))

    for rule_idx in range(anfis_model.n_rules):
        c   = centres[rule_idx]
        s   = sigmas[rule_idx]
        mfv = np.exp(-0.5 * ((x_range - c) / (s + 1e-8)) ** 2)
        ax.plot(x_range, mfv, color=colors[rule_idx], alpha=0.55, lw=1.5)

    ax.set_xlabel('PCA Component 1', fontsize=10)
    ax.set_ylabel('Membership Degree', fontsize=10)
    ax.set_title(
        f'Learned Membership Functions — {class_name}\n({anfis_model.n_rules} fuzzy rules)',
        fontsize=11, fontweight='bold'
    )
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# ==============================================================================
# Disease information database
# ==============================================================================

DISEASE_INFO = {
    "Ringworm": {
        "description": "A fungal infection of the skin caused by dermatophytes. "
                       "Appears as a ring-shaped, scaly rash.",
        "symptoms"   : "Circular rash, itching, scaly patches, hair loss in affected area.",
        "note"       : "Highly contagious. Antifungal creams are the standard treatment."
    },
    "Psoriasis": {
        "description": "A chronic autoimmune skin condition that speeds up the skin "
                       "cell lifecycle, causing cells to build up rapidly on the surface.",
        "symptoms"   : "Red patches, silvery scales, dry/cracked skin, itching, burning.",
        "note"       : "Not contagious. Managed with topical treatments and systemic therapies."
    },
    "Athlete's Foot": {
        "description": "A fungal infection primarily affecting the feet, especially "
                       "between the toes. Also known as tinea pedis.",
        "symptoms"   : "Itching, burning, blistering between toes, peeling skin.",
        "note"       : "Keep feet dry and clean. Over-the-counter antifungals are effective."
    },
    "Eczema": {
        "description": "A condition that makes skin red and itchy. Often found in "
                       "children but can occur at any age.",
        "symptoms"   : "Dry skin, itching, red patches, small raised bumps, thickened skin.",
        "note"       : "Triggers include stress, irritants, and allergens. Moisturisers help."
    },
    "No Skin Detected": {
        "description": "The uploaded image does not appear to show a skin region "
                       "clearly visible to the detection system.",
        "symptoms"   : "N/A",
        "note"       : "Please upload a clear, close-up image of the affected skin area."
    },
    "Normal": {
        "description": "No skin disease detected. The image shows what appears to "
                       "be healthy, unaffected skin.",
        "symptoms"   : "N/A",
        "note"       : "If you have concerns, consult a dermatologist for a professional assessment."
    }
}


# ==============================================================================
# Main application
# ==============================================================================

def main():
    inject_custom_css()

    # ---- Sidebar ----
    with st.sidebar:
        st.title("Skin Disease Detector")
        st.markdown("---")
        st.markdown("**About this system**")
        st.markdown(
            "This tool uses a **CNN-ANFIS hybrid** model:\n"
            "- **EfficientNetV2M** extracts visual features from your image\n"
            "- **ANFIS** (Adaptive Neuro-Fuzzy Inference System) classifies "
            "those features using learnable fuzzy logic rules\n"
            "- A **one-vs-rest** scheme produces a confidence score per class"
        )
        st.markdown("---")
        st.markdown("**Detectable conditions**")
        for name in DISEASE_INFO:
            st.markdown(f"  - {name}")
        st.markdown("---")
        st.markdown("**Model Info**")
        st.markdown("Backbone: EfficientNetV2M (ImageNet)")
        st.markdown("Classifier: ANFIS (Takagi-Sugeno, OvR)")
        st.markdown("Feature dim: 1280 -> PCA -> ANFIS")
        st.markdown("---")
        show_mf_plot = st.checkbox("Show membership function plot", value=True)
        show_raw_scores = st.checkbox("Show raw confidence table", value=False)

    # ---- Page header ----
    st.markdown(
        "<h1 style='color:#1a5276;'>Skin Disease Detection</h1>"
        "<p style='color:#555; font-size:1.05rem;'>"
        "CNN-ANFIS Neuro-Fuzzy Hybrid System &mdash; Bahria University Islamabad"
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ---- Load models ----
    models_available = ANFIS_CONFIG_PATH.exists() and ANFIS_STATES_PATH.exists()

    if not models_available:
        st.error(
            "Model files not found in the `models/` directory. "
            "Please run both training notebooks and copy the following files "
            "into `models/` before deploying:\n"
            "- `anfis_config.pkl`\n"
            "- `cnn_anfis_all_states.pt`\n"
            "- `pca_cnn.pkl`\n"
            "- `scaler_cnn.pkl`"
        )
        st.stop()

    anfis_bundle   = load_anfis_models()
    backbone_model = load_feature_extractor()
    class_names    = anfis_bundle['config']['class_names']

    # ---- Image upload ----
    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        st.markdown("<div class='section-header'>Upload Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a clear, close-up image of the affected skin area.",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed"
        )
        st.markdown(
            "<div class='disclaimer'>"
            "<strong>Medical disclaimer:</strong> This tool is intended for educational "
            "and research purposes only. It is not a substitute for professional medical "
            "diagnosis. Always consult a qualified dermatologist for medical advice."
            "</div>",
            unsafe_allow_html=True
        )

    with col_preview:
        if uploaded_file is not None:
            st.markdown("<div class='section-header'>Uploaded Image</div>", unsafe_allow_html=True)
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, use_column_width=True)

    if uploaded_file is None:
        st.info("Please upload a skin image to begin analysis.")
        return

    # ---- Run inference ----
    st.markdown("---")
    st.markdown("<div class='section-header'>Analysis Results</div>", unsafe_allow_html=True)

    with st.spinner("Analysing image ..."):
        raw_features = extract_features_from_image(pil_image, backbone_model)
        predicted_idx, conf_scores, is_ambiguous = run_anfis_inference(
            raw_features, anfis_bundle
        )

    predicted_class_name = class_names[predicted_idx]
    top_confidence       = float(conf_scores[predicted_idx])

    # ---- Primary result card ----
    col_result, col_chart = st.columns([1, 1])

    with col_result:
        st.markdown(
            f"<div class='result-card'>"
            f"<p class='prediction-label'>{predicted_class_name}</p>"
            f"<p class='confidence-text'>Confidence: {top_confidence:.1%}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Ambiguity warning
        if is_ambiguous:
            second_idx   = int(np.argsort(conf_scores)[::-1][1])
            second_name  = class_names[second_idx]
            second_score = float(conf_scores[second_idx])
            st.markdown(
                f"<div class='warning-card'>"
                f"<strong>Ambiguous prediction</strong><br>"
                f"The model is uncertain between "
                f"<em>{predicted_class_name}</em> ({top_confidence:.1%}) and "
                f"<em>{second_name}</em> ({second_score:.1%}). "
                f"A dermatologist's assessment is strongly recommended."
                f"</div>",
                unsafe_allow_html=True
            )

        # Disease information
        info = DISEASE_INFO.get(predicted_class_name, {})
        if info:
            with st.expander("About this condition", expanded=True):
                st.markdown(f"**Description:** {info.get('description', '')}")
                st.markdown(f"**Common symptoms:** {info.get('symptoms', '')}")
                st.markdown(f"**Note:** {info.get('note', '')}")

    with col_chart:
        confidence_fig = render_confidence_bar_chart(conf_scores, class_names, predicted_idx)
        st.pyplot(confidence_fig)

    # ---- Membership function plot (interpretability) ----
    if show_mf_plot:
        st.markdown("---")
        st.markdown("<div class='section-header'>Fuzzy Rule Interpretability</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "The chart below shows the learned **Gaussian membership functions** "
            "for the predicted class on PCA Component 1. Each curve is one fuzzy rule. "
            "This is what makes ANFIS interpretable — unlike a softmax classifier, "
            "you can see how the model partitions the feature space."
        )

        pca_range_min = -4.0
        pca_range_max =  4.0
        mf_fig = render_membership_function_plot(
            anfis_bundle['models'][predicted_idx],
            predicted_class_name,
            (pca_range_min, pca_range_max)
        )
        st.pyplot(mf_fig)

    # ---- Raw confidence scores table ----
    if show_raw_scores:
        st.markdown("---")
        st.markdown("<div class='section-header'>Raw Confidence Scores</div>",
                    unsafe_allow_html=True)
        score_df = {
            'Class'            : class_names,
            'Confidence Score' : [f"{s:.4f}" for s in conf_scores],
            'Predicted'        : ["Yes" if i == predicted_idx else "No"
                                  for i in range(len(class_names))]
        }
        st.dataframe(score_df, use_container_width=True)

    # ---- Footer ----
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>"
        "Neural Networks &amp; Fuzzy Logic &nbsp;|&nbsp; Bahria University Islamabad "
        "&nbsp;|&nbsp; Ali Ahmed Malik &amp; Shaheer Tariq"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
