import tensorflow as tf
import mediapipe as mp
import numpy as np
import cv2


def detectFaces(image, face_detection):
    # Mark image as not writeable for improved performance & convert to RGB
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the faces & return the result
    return face_detection.process(image)


def eyeIsClosed(faces, image, model):
    # If no faces are detected, return False
    if not faces.detections:
                return False

    # Extract every face from the image & detect whether their eyes er open or closed
    for face in faces.detections:
        
        # Extract the eyes from the image
        eyes = extractEyes(image, face)

        # Reshape the eyes for the model
        eyes = [eye.reshape(1, 40, 40, 3) for eye in eyes]

        # Predict the state of the eyes using a CNN
        right_state = model.predict(eyes[0])
        left_state = model.predict(eyes[1])

        # If any eyes are closed, return True
        if np.argmax(left_state) == 0 or np.argmax(right_state) == 0:
            return True

    # If all eyes are open, return False
    return False


def extractEyes(image, data):
    keypoints = data.location_data.relative_keypoints
    # Find the left and right eye coordinates
    (leye_x, leye_y) = int(keypoints[0].x*640), int(keypoints[0].y*480)
    (reye_x, reye_y) = int(keypoints[1].x*640), int(keypoints[1].y*480)

    # Extract the eyes from the image
    leye = image[leye_y-20:leye_y+20, leye_x-20:leye_x+20]
    reye = image[reye_y-20:reye_y+20, reye_x-20:reye_x+20]

    return [leye, reye]


def main():
    with mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.7) as face_detection:
        # Load eye state CNN model
        model = tf.keras.models.load_model('models/eye_state')

        count = 1
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            start_time = time.time()   
            ret, frame = cap.read()
            # Check if the frame is empty
            if not ret:
                continue
            
            # Detect faces in the frame
            faces = detectFaces(frame, face_detection)
            
            #If any eyes are closed, save the terrible photo :)
            if eyeIsClosed(faces, frame, model):
                cv2.imwrite(f'images/unusable_image{count}.jpg', frame)
                count += 1

        # Close videocapture after usage
        cap.release()


if __name__ == "__main__":
    main()