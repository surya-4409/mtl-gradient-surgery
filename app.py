import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import tensorflow as tf
from src.dataset import get_datasets
from src.models import MultiTaskModel

# Streamlit Page Config
st.set_page_config(page_title="MTL Gradient Surgery Monitor", layout="wide")
st.title("Multi-Task Learning: PCGrad & Gradient Conflict Monitor")

# --- Helper to load data ---
@st.cache_data
def load_metrics():
    try:
        df_base = pd.read_csv('results/baseline_metrics.csv')
        df_pc = pd.read_csv('results/pcgrad_metrics.csv')
        df_conflict = pd.read_csv('results/gradient_conflict.csv')
        with open('results/final_metrics.json', 'r') as f:
            final_metrics = json.load(f)
        return df_base, df_pc, df_conflict, final_metrics
    except FileNotFoundError:
        st.error("Metrics files not found. Please run the training scripts first.")
        return None, None, None, None

df_base, df_pc, df_conflict, final_metrics = load_metrics()

if df_base is not None:
    # --- Tabbed Layout ---
    tab1, tab2, tab3 = st.tabs(["Gradient Conflict Monitor", "Task Performance Dashboard", "Shared Representation Inspector"])

    # --- TAB 1: Gradient Conflict Monitor ---
    with tab1:
        st.header("Gradient Cosine Similarity Over Time")
        # Wrapper div with the required data-testid
        st.markdown('<div data-testid="gradient-conflict-monitor">', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot similarity
        ax.plot(df_conflict['step'], df_conflict['cosine_similarity'], color='blue', alpha=0.7, label='Cosine Similarity')
        
        # Highlight conflicts (values < 0) in red
        conflicts = df_conflict[df_conflict['cosine_similarity'] < 0]
        ax.scatter(conflicts['step'], conflicts['cosine_similarity'], color='red', s=10, label='Conflict (< 0)')
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Gradient Conflict Detection (Shared Backbone)')
        ax.legend()
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: Task Performance Dashboard ---
    with tab2:
        st.header("Baseline vs. PCGrad Performance")
        st.markdown('<div data-testid="performance-dashboard">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Task A Validation Accuracy")
            fig_a, ax_a = plt.subplots(figsize=(8, 4))
            ax_a.plot(df_base['epoch'], df_base['val_acc_a'], label='Baseline', marker='o')
            ax_a.plot(df_pc['epoch'], df_pc['val_acc_a'], label='PCGrad', marker='s')
            ax_a.set_xlabel('Epoch')
            ax_a.set_ylabel('Accuracy')
            ax_a.legend()
            st.pyplot(fig_a)

        with col2:
            st.subheader("Task B Validation Accuracy")
            fig_b, ax_b = plt.subplots(figsize=(8, 4))
            ax_b.plot(df_base['epoch'], df_base['val_acc_b'], label='Baseline', marker='o')
            ax_b.plot(df_pc['epoch'], df_pc['val_acc_b'], label='PCGrad', marker='s')
            ax_b.set_xlabel('Epoch')
            ax_b.set_ylabel('Accuracy')
            ax_b.legend()
            st.pyplot(fig_b)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 3: Shared Representation Inspector ---
    with tab3:
        st.header("UMAP Projection of Shared Representation Space")
        st.markdown('<div data-testid="representation-inspector">', unsafe_allow_html=True)
        st.write("Generating representations from the validation set (this may take a moment)...")
        
        @st.cache_resource
        def generate_umap_projections():
            # Get a batch of validation data to inspect
            _, val_dataset = get_datasets(batch_size=1000)
            for inputs, (labels_a, labels_b) in val_dataset.take(1):
                X_sample = inputs
                y_a_sample = labels_a.numpy()
                y_b_sample = labels_b.numpy()
            
            # Initialize model and pass data through backbone
            model = MultiTaskModel()
            _ = model(X_sample) # Build model weights
            shared_reps = model.backbone(X_sample).numpy()
            
            # Apply UMAP
            reducer = umap.UMAP(random_state=42)
            embedding = reducer.fit_transform(shared_reps)
            return embedding, y_a_sample, y_b_sample

        embedding, y_a, y_b = generate_umap_projections()
        
        col_u1, col_u2 = st.columns(2)
        
        with col_u1:
            st.subheader("Colored by Task A Labels")
            fig_u1, ax_u1 = plt.subplots(figsize=(6, 6))
            scatter1 = ax_u1.scatter(embedding[:, 0], embedding[:, 1], c=y_a, cmap='coolwarm', s=15, alpha=0.8)
            plt.colorbar(scatter1, ax=ax_u1)
            st.pyplot(fig_u1)
            
        with col_u2:
            st.subheader("Colored by Task B Labels")
            fig_u2, ax_u2 = plt.subplots(figsize=(6, 6))
            scatter2 = ax_u2.scatter(embedding[:, 0], embedding[:, 1], c=y_b, cmap='viridis', s=15, alpha=0.8)
            plt.colorbar(scatter2, ax=ax_u2)
            st.pyplot(fig_u2)
            
        st.markdown('</div>', unsafe_allow_html=True)