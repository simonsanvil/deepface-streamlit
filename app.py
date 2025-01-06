import streamlit as st 
from deepface import DeepFace
import tempfile
from PIL import Image, ImageDraw

def main():
    st.set_page_config(
        page_title="DeepFace Demo",
        page_icon=":sunglasses:",
        # layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("DeepFace Demo! :sunglasses:")
    st.write("This app uses [DeepFace](https://github.com/serengil/deepface) to analyze your face and display the results.")
    st.markdown("### Face Input")
    # check if the session state has the uploaded file
    camera_input = st.session_state.get("camera_input")
    uploaded_input = st.session_state.get("upload_input")
    if uploaded_file:=(camera_input or uploaded_input):
        st.image(uploaded_file, caption="Your picture", width=300)
        if st.button("Reset", key="reset", type="primary"):
            st.session_state.clear()
            st.rerun()
    else:
        # get the uploaded file or camera input
        tabs = st.tabs(["Upload", "Camera"])
        with tabs[0]:
            uploaded_input = st.file_uploader("Upload a picture :)", type=["jpg", "jpeg", "png"], key="upload_input")
        with tabs[1]:
            camera_input = st.camera_input("Take a selfie :)", key="camera_input")
    if camera_input and uploaded_input:
        st.warning("Please use either the camera or upload a picture, not both.")
        st.stop()
    if camera_input:
        uploaded_file = camera_input
    else:
        uploaded_file = uploaded_input
    if not uploaded_file:
        st.warning("Please upload a picture or use the camera.")
        st.stop()
    # analyze the input image with DeepFace
    with st.spinner("Analyzing image..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                result = DeepFace.analyze(
                    img_path = tmp_file.name,
                    actions = ['age', 'gender', 'race', 'emotion'],
                )    
                pil_image = Image.open(tmp_file.name)
        except Exception as e:
            st.error(f"Error Analyzing image: {e}")
            st.stop()
        result_dict = result[0]
        age = result_dict['age']
        gender = result_dict['dominant_gender']
        emotion = result_dict['dominant_emotion']
        race = result_dict['dominant_race']
        st.markdown(
            "### Results\n\n"
            "| Age | Gender | Emotion | Race |\n"
            "| --- | ------ | ------- | ---- |\n"
            f"| {str(age).title()} | {str(gender).title()} | {str(emotion).title()} | {str(race).title()} |\n"
        )
        with st.expander("Full Results"):
            # draw a rectangle around the detected face
            img_bbox = result_dict['region']
            draw = ImageDraw.Draw(pil_image)
            draw.rectangle([(img_bbox['x'], img_bbox['y']), (img_bbox['x'] + img_bbox['w'], img_bbox['y'] + img_bbox['h'])], outline="red", width=5)
            st.image(pil_image, caption="Detected face", width=200)
            st.write("DeepFace Results:")
            st.write(result)


if __name__ == "__main__":
    main()