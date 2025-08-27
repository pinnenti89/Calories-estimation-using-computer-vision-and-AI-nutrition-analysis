import streamlit as st
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from segmentation import segment_image
from classification import load_model, get_transform, classify_images
from nutrition import estimate_weights_from_masks, estimate_nutrition
from depth_estimation import estimate_depth

# Ensure set_page_config is the first Streamlit command
st.set_page_config(page_title="🍽️ Food Analyzer", layout="wide")

def main():
    # Custom CSS for enhanced UI (added after set_page_config)
    st.markdown(
        """
        <style>
        /* Background with acrylic painting-inspired gradient */
        .stApp {
            background: linear-gradient(135deg, #e0e7ff, #f9e4e4, #f0f0f0);
            background-size: cover;
            background-attachment: fixed;
        }
        /* Header styling */
        .css-1d391kg {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        /* Sidebar styling */
        .css-1a32cs0 {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 10px;
            box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
        }
        /* Section headers */
        .css-1v3fvcr {
            color: #2c3e50;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
        }
        /* Button styling */
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #2980b9;
            transform: scale(1.05);
        }
        /* Image container */
        .stImage {
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🍽️ Food Analyzer")
    st.markdown("Upload an image of food to analyze its components, classify items, and estimate nutrition!")

    # Sidebar
    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("Model Path", "indian_food_classifier.pth")
    nutrition_file = st.sidebar.text_input("Nutrition File", r"C:\Users\monis\Downloads\food_item_macros.xlsx")
    total_weight = st.sidebar.number_input("Total Weight (grams)", 100, 2000, 1000)
    k_value = st.sidebar.number_input("Number of Segments (k)", min_value=2, max_value=10, value=4)

    # Main content
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getvalue())

        col1, col2 = st.columns(2)
        
        # Segmentation
        with col1:
            st.subheader("Segmentation Results")
            # Pass the user-specified k value here
            segmented_img, detected_img, crop_paths = segment_image("temp_image.png", k=k_value)
            st.image(segmented_img, caption="Segmented Image", use_column_width=True)
            st.image(detected_img, caption="Detected Items", use_column_width=True)

        # Classification
        with col2:
            st.subheader("Classified Food Items")
            model = load_model(model_path)
            transform = get_transform()
            predictions = classify_images(model, transform, "food_segments")
            
            for orig_name, label, new_path in predictions:
                st.image(new_path, caption=f"{orig_name} → {label}", width=200)

        # Nutrition Analysis
        st.subheader("Nutrition Analysis")
        weights = estimate_weights_from_masks("project/masks", total_weight)
        nutrition_data, totals = estimate_nutrition(weights, nutrition_file)
        
        # Display nutrition table
        st.table(nutrition_data)
        
        # Display totals
        total_cals, total_carbs, total_prots, total_fats = totals
        st.write(f"**Total Nutrition:**")
        st.write(f"Calories: {total_cals:.1f} kcal")
        st.write(f"Carbohydrates: {total_carbs:.1f} g")
        st.write(f"Proteins: {total_prots:.1f} g")
        st.write(f"Fats: {total_fats:.1f} g")

        # Pie chart
        fig, ax = plt.subplots()
        labels = ['Carbohydrates', 'Proteins', 'Fats']
        values = [total_carbs * 4, total_prots * 4, total_fats * 9]
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999', '#99ff99', '#9999ff'])
        ax.axis('equal')
        st.pyplot(fig)

        # Depth Estimation
        st.subheader("Depth Estimation")
        depth_map = estimate_depth("temp_image.png")
        fig, ax = plt.subplots()
        ax.imshow(depth_map, cmap="inferno")
        ax.axis("off")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
