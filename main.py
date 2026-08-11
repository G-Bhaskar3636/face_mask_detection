import streamlit as st
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

model = load_model("mask_detector.keras")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.title("😷 Face Mask Detection")
st.write("Use your camera to detect whether a person is wearing a mask.")

camera_image = st.camera_input("Take a picture")

if camera_image is not None:

    # Convert uploaded image to OpenCV format
    image_bytes = camera_image.getvalue()
    image_array = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Detect faces on the ORIGINAL image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:

        st.warning("No face detected.")

    else:

        st.success(f"{len(faces)} face(s) detected.")

        # Process each detected face
        for (x, y, w, h) in faces:

            # Crop the face
            face = image[y:y+h, x:x+w]

            # Resize face for VGG16
            face = cv2.resize(face, (224, 224))

            # Add batch dimension
            face = np.expand_dims(face, axis=0)

            # VGG16 preprocessing
            face = preprocess_input(face)

            # Prediction
            prediction = model.predict(face, verbose=0)

            # Convert prediction to class
            if prediction[0][0] > 0.5:
                result = "Without Mask"
            else:
                result = "With Mask"

            # Draw rectangle
            cv2.rectangle(
                image,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            # Display result
            cv2.putText(
                image,
                result,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Convert BGR → RGB for Streamlit
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(
            image_rgb,
            caption="Mask Detection Result",
            use_container_width=True
        )
