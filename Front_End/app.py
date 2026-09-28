
from PIL import Image
import streamlit as st
import requests



logo = Image.open(
    "D:/Thermal surrogate project/assets/si-vision-logo (2).png"
)

st.set_page_config(
    page_title="Si-Vision Thermal Prediction",
    page_icon=logo,
    layout="wide"
)




st.image(
    "D:/Thermal surrogate project/assets/si-vision-logo (2).png",
    width=180
)

st.title("3D-IC Thermal Prediction")

st.write(
    "Upload your input files and run the thermal model."
)




st.subheader("📁 Upload Input Files")

files = st.file_uploader(
    "Drag & Drop your files here",
    type=["stk", "flp"],
    accept_multiple_files=True
)




if files:

    st.write("Uploaded files:")

    for file in files:
        st.write("📄", file.name)




if st.button("▶ Run Thermal Model"):

    if not files:

        st.warning("Please upload your files first.")

    else:

        with st.spinner("Running Thermal Model..."):

            try:

                # Prepare files
                uploaded_files = []

                for file in files:

                    uploaded_files.append(
                        (
                            "Files",
                            (
                                file.name,
                                file.getvalue(),
                                "application/octet-stream"
                            )
                        )
                    )

                # Send files to FastAPI
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files=uploaded_files
                )



                if response.status_code == 200:

                    result = response.json()

                    st.success(
                        "✅ Model finished successfully!"
                    )



                    st.subheader("🌡️ Thermal Prediction")

                    image_url = ("http://127.0.0.1:8000/plot/" + result["plot_name"])

                    st.image(image_url , use_container_width=True)



                    st.subheader("📊 Output")

                    st.write(
                        "Number of nodes:",
                        result["num_nodes"]
                    )

                    st.write(
                        "Minimum Temperature:",
                        f'{result["min_temperature_K"]:.2f} K'
                    )

                    st.write(
                        "Maximum Temperature:",
                        f'{result["max_temperature_K"]:.2f} K'
                    )

                    st.write(
                        "Mean Temperature:",
                        f'{result["mean_temperature_K"]:.2f} K'
                    )



                    st.subheader("🌡️ Node Temperatures")

                    st.write(result["temperatures_K"])



                else:

                    st.error(f"Backend Error: {response.status_code}")

                    st.code(response.text)

            # ========================================================
            # CONNECTION ERROR
            # ========================================================

            except Exception as e:

                st.error(f"Connection Error: {e}")