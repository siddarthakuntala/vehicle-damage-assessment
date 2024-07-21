import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from fpdf import FPDF

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\sidda\OneDrive\Desktop\project dataset\my_model.h5')

# Class labels for the vehicle damage types
class_labels = ['scratch', 'crack', 'tire flat', 'dent', 'shattered window', 'broken headlamp']

# Estimated cost associated with each type of damage
damage_costs = {
    'scratch': 100,
    'crack': 300,
    'tire flat': 150,
    'dent': 500,
    'shattered window': 700,
    'broken headlamp': 400
}

def predict_top_3_damages(img_array, model, class_labels):
    # Make predictions
    predictions = model.predict(img_array)
    predictions = predictions[0]  # Get the first (and only) prediction array

    # Get the top 3 predictions
    top_3_indices = np.argsort(predictions)[-3:][::-1]
    top_3_labels = [class_labels[i] for i in top_3_indices]
    top_3_scores = [predictions[i] for i in top_3_indices]

    return list(zip(top_3_labels, top_3_scores))

def estimate_repair_cost(damages, damage_costs):
    total_cost = 0
    for label, score in damages:
        total_cost += damage_costs.get(label, 0) * score
    return total_cost

def create_pdf(report_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Insurance Claim Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Name: {report_data['name']}", ln=True)
    pdf.cell(200, 10, txt=f"Email: {report_data['email']}", ln=True)
    pdf.cell(200, 10, txt=f"Phone: {report_data['phone']}", ln=True)
    pdf.cell(200, 10, txt=f"Damage Predictions:", ln=True)
    
    for label, score in report_data['damages']:
        pdf.cell(200, 10, txt=f"{label}: {score:.4f}", ln=True)
    
    pdf.cell(200, 10, txt=f"Estimated Repair Cost: ${report_data['cost']:.2f}", ln=True)
    
    return pdf

st.title("Vehicle Damage Assessment")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image if your model expects normalized input

    # Display the uploaded image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict top 3 damages
    top_3_damages = predict_top_3_damages(img_array, model, class_labels)

    # Display the results
    st.write("Prediction Results:")
    for label, score in top_3_damages:
        st.write(f"{label}: {score:.4f}")

    # Estimate repair cost
    estimated_cost = estimate_repair_cost(top_3_damages, damage_costs)
    st.write(f"Estimated Repair Cost: ${estimated_cost:.2f}")

    # Insurance claim report form
    st.write("### Insurance Claim Report")
    with st.form("claim_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        submit = st.form_submit_button("Generate Report")

    if submit:
        report_data = {
            "name": name,
            "email": email,
            "phone": phone,
            "damages": top_3_damages,
            "cost": estimated_cost
        }
        pdf = create_pdf(report_data)
        
        pdf_output_path = "insurance_claim_report.pdf"
        pdf.output(pdf_output_path)
        
        with open(pdf_output_path, "rb") as pdf_file:
            st.download_button(
                label="Download Report",
                data=pdf_file,
                file_name="insurance_claim_report.pdf",
                mime="application/octet-stream"
            )
