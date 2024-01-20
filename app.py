import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import tensorflow as tf
from utils import mediapipe_detection, extract_keypoints, image_resize, mp_holistic


sequence = []
sentence = []
predictions = []
threshold = 0.975
actions = np.array(['hello', 'thanks', 'iloveyou', 'yes', 'no', 'mother', 'father'])
model = tf.keras.models.load_model('model/model_hands_12.h5')


st.title('Sign Language Detector Application using MediaPipe')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Sign Language Detector Application using MediaPipe')
st.sidebar.subheader('Parameters')


app_mode = st.sidebar.selectbox('Choose the App mode',
['About App','Run on Webcam','Upload a Video']
)


if app_mode =='About App':
    st.markdown('In this application we are using **MediaPipe** for live analysis of hand signs. **StreamLit** is to create the Web Graphical User Interface (GUI) ')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    st.markdown('''
          # About Me \n 
            Hey this is **Akshat Bakshi**. \n
            I am a student in Thapar institute of engineering and technology, persuing B.E. in Electronics and communications engeneering . \n
            
            You can find the code on this github repository:
            [HERE]()             
            ''')


elif app_mode =='Run on Webcam':

    use_webcam = st.sidebar.checkbox('Use Webcam')
    reset_webcam = st.sidebar.button('Reset Webcam')
    
    if reset_webcam:
        st.experimental_rerun()

    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
    unsafe_allow_html=True,
        )

    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0, max_value = 1.0, value = 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0, max_value = 1.0, value = 0.5)

    st.sidebar.markdown('---')

    st.markdown(' ## Output')

    stframe = st.empty()
    
    if use_webcam:
        vid = cv2.VideoCapture(0)
    else:
        st.warning("Please check 'Use Webcam' to run the app with the webcam.")
        vid = None
    
    if vid is not None:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = 30
        fps = 0
        i = 0

        last_predicted_action = "No action"
        
        kpi1, kpi2, kpi3 = st.columns(3)
        
        with kpi1:
            st.markdown("<h1 style='text-align: center; color: white;'>FrameRate</h1>", unsafe_allow_html=True)
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("<h1 style='text-align: center; color: white;'>Prediction</h1>", unsafe_allow_html=True)
            kpi2_text = st.markdown(f"<h1 style='text-align: center; color: red;'>{last_predicted_action}</h1>", unsafe_allow_html=True)

        with kpi3:
            st.markdown("<h1 style='text-align: center; color: white;'>Width</h1>", unsafe_allow_html=True)
            kpi3_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)
        

        with mp_holistic.Holistic( min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence ) as holistic:
            prevTime = 0
            
            while vid.isOpened():
                i +=1
                ret, frame = vid.read()
                if not ret:
                    continue

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                
                # Draw landmarks
                # draw_landmarks(image, results)
                
                # Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))
                    
                    if np.unique(predictions[-10:])[0]==np.argmax(res):
                        
                        if res[np.argmax(res)] > threshold: 
                            
                            if len(sentence) > 0:
                                
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]
                    
                last_predicted_action = sentence[-1] if sentence else "No action"
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
                
                # Dashboard
                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{last_predicted_action}</h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

                frame = cv2.resize(frame,(0,0), fx = 0.8 , fy = 0.8)
                frame = image_resize(image = frame, width = 640)
                stframe.image(frame, channels = 'BGR', use_column_width=True)

        st.text('Video Processed')
        
        vid.release()

    else:
        st.warning("Webcam not available. Please check 'Use Webcam' to run the app with the webcam or try using the 'Reset Webcam' button to re-run the app.")


elif app_mode == 'Upload a Video':
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"])

    if uploaded_file is not None:
        st.sidebar.markdown('---')

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tfflie:
            tfflie.write(uploaded_file.read())

        # Process the uploaded video
        uploaded_vid = cv2.VideoCapture(tfflie.name)
        width = int(uploaded_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(uploaded_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(uploaded_vid.get(cv2.CAP_PROP_FPS))
        stframe = st.empty()

        detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
        tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)

        kpi2_uploaded, kpi3_uploaded = st.columns(2)

        last_predicted_action = "No action"
        
        with kpi2_uploaded:
            st.markdown("<h1 style='text-align: center; color: white;'>Prediction</h1>", unsafe_allow_html=True)
            kpi2_text = st.markdown(f"<h1 style='text-align: center; color: red;'>{last_predicted_action}</h1>", unsafe_allow_html=True)

        with kpi3_uploaded:
            st.markdown("<h1 style='text-align: center; color: white;'>Width</h1>", unsafe_allow_html=True)
            kpi3_text = st.markdown(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

        with mp_holistic.Holistic(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence) as holistic:
            while uploaded_vid.isOpened():
                ret, frame = uploaded_vid.read()
                if not ret:
                    break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                # draw_landmarks(image, results)

                # Prediction logic (replace this with your existing logic)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))

                    if np.unique(predictions[-10:])[0] == np.argmax(res):

                        if res[np.argmax(res)] > threshold:

                            if len(sentence) > 0:

                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                last_predicted_action = sentence[-1] if sentence else "No action"
               
                # Display the processed frame
                frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                frame = image_resize(image=frame, width=640)
                stframe.image(frame, channels='BGR', use_column_width=True)

                # Update KPIs
                kpi2_text.markdown(f"<h1 style='text-align: center; color: red;'>{last_predicted_action}</h1>", unsafe_allow_html=True)
                kpi3_text.markdown(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

        uploaded_vid.release()
        st.text('Video Processed')
