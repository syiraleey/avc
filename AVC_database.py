import mysql.connector
import streamlit as st
import cv2
from ultralytics import YOLO
import math
import tempfile

# Load YOLO model
model = YOLO("../project fyp/best (1).pt")

# Class names
classNames = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Alert']

# Toll plaza information
toll_plazas = {
    "Batu 9": {
        'Class 0': 'Free',
        'Class 1': 'RM 1.30',
        'Class 2': 'RM 2.60',
        'Class 3': 'RM 2.60',
        'Class 4': 'RM 0.70',
        'Class 5': 'RM 1.00',
        'Alert': 'Warning!'
    },
    "Batu 11": {
        'Class 0': 'Free',
        'Class 1': 'RM 1.30',
        'Class 2': 'RM 2.60',
        'Class 3': 'RM 2.60',
        'Class 4': 'RM 0.70',
        'Class 5': 'RM 0.90',
        'Alert': 'Warning!'
    }
}

#Establish a connection to MySQL Server
mydb = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    database="avc_system"
)

mycursor = mydb.cursor()
print("Connection Established")

# CREATE STREAMLIT APP

def main():
    # Initialize Streamlit app
    st.title("Automated Vehicle Classification (AVC) for Malaysia's Toll System")

    # Create a placeholder for the video
    video_placeholder = st.empty()

    # Create a placeholder for displaying detected class and toll fare
    detected_class_placeholder = st.empty()

    # Create a sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        # Add a spacer for better readability
        st.text("")
        st.subheader("Lebuhraya Cheras - Kajang (GRANDSAGA)")

        # Choose toll plaza
        selected_toll_plaza = st.selectbox("Choose Toll Plaza", list(toll_plazas.keys()))

        # Choose between video upload and webcam
        use_webcam = st.checkbox("Use Camera")
        if not use_webcam:
            # File upload
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
        else:
            uploaded_file = None

    # Check if a file is uploaded or webcam is chosen
    if uploaded_file is not None or use_webcam:
        if use_webcam:
            # OpenCV VideoCapture for webcam
            cap = cv2.VideoCapture(0)
        else:
            # Create a temporary file to store the uploaded video
            with tempfile.NamedTemporaryFile(delete=False) as temp_video:
                temp_video.write(uploaded_file.read())
                temp_video_path = temp_video.name

            # Load the uploaded video
            cap = cv2.VideoCapture(temp_video_path)

        # Create a placeholder for the detected class and toll fare box
        detection_box = st.empty()

        # Initialize variables for tracking the current vehicle
        current_vehicle = None
        vehicle_disappeared = False

        while True:
            # Read a frame from the video
            success, img = cap.read()

            # Check if the video has ended
            if not success:
                break

            # If there's a tracked vehicle, process it
            if current_vehicle:
                x1, y1, x2, y2 = current_vehicle["bbox"]

                # Draw bounding box on the original video frame
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Calculate width and height of the bounding box
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = current_vehicle["conf"]

                # Class Name
                detected_class = classNames[current_vehicle["cls"]]

                # Display the class and confidence slightly above the bounding box
                text_position = (max(0, x1), max(0, y1 - 30))
                toll_info = toll_plazas[selected_toll_plaza][detected_class]
                cv2.putText(img, f'{detected_class} {conf}', text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

                # Update the detected class text
                detected_class_text = f"Class: {detected_class}<br>Toll Fare: {toll_info}"

                # Add a bit of gap between the boxes
                st.text("")  # Add an empty space or adjust as needed

                sql = "insert into vehicleclassdetection(vehicleClass, TollFare) values(%s, %s)"
                val = (detected_class, toll_info)
                mycursor.execute(sql, val)
                mydb.commit()
                st.success("Record Inserted Successfully")

                # Update the detected class and toll fare box
                detection_box.markdown(
                    f'<div style="border: 2px solid #FF00FF; padding: 10px; border-radius: 5px; background-color: #f4f4f4;">'
                    f'<p style="font-size: 16px; margin: 0;">{detected_class_text}</p>'
                    f'</div>', unsafe_allow_html=True)

                # Check if the current vehicle is starting to disappear
                if x2 < img.shape[1] * 0.9:
                    vehicle_disappeared = True

                # If the vehicle has disappeared, reset the current vehicle
                if vehicle_disappeared:
                    current_vehicle = None
                    vehicle_disappeared = False

            else:
                # Use the YOLO model to detect vehicles
                results = model(img, stream=True)

                # List to store the detected vehicles
                detected_vehicles = []

                # Iterate over the results
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100

                        # Class Name
                        cls = int(box.cls[0])

                        # Add the detected vehicle to the list
                        detected_vehicles.append({
                            "bbox": (x1, y1, x2, y2),
                            "conf": conf,
                            "cls": cls })

                # If there are detected vehicles, select the frontmost one
                if detected_vehicles:
                    # Sort the list based on the x-coordinate of the bounding boxes
                    detected_vehicles.sort(key=lambda v: v["bbox"][0])

                    # Select the frontmost vehicle
                    current_vehicle = detected_vehicles[0]


            # Display the video with bounding boxes in the Streamlit app
            video_placeholder.image(img, channels="BGR", use_column_width=True)

        # Release the video capture object
        cap.release()
        if not use_webcam:
            # Remove the temporary video file
            st.experimental_rerun()  # Rerun the Streamlit app to remove the temporary video file

if __name__ == "__main__":
    main()