import joblib
import json
import numpy as np
import base64
from wavelet import w2d
import cv2

_class_name_to_number = {}
_class_number_to_name = {}

_model = None

def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    result = []
    print(len(imgs))
    for img in imgs:
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32 * 32 * 3, 1), scaled_img_har.reshape(32 * 32, 1)))
        len_image_array = 32 * 32 * 3 + 32 * 32

        final = combined_img.reshape(1,len_image_array).astype(float)
        result.append({
            'class':_class_number_to_name[_model.predict(final)[0]],
            "class_probablity":np.around(_model.predict_proba(final)*100,2).tolist()[0], #get probablities of each class
            'class_dictionary':_class_name_to_number
        })
    return result

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(",")[1]
    nparr = np.frombuffer(base64.b64encode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped_faces = []

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes)>=2:
            cropped_faces.append(roi_color)

    return cropped_faces

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global _class_name_to_number
    global _class_number_to_name
    global _model

    with open("./artifacts/cele_face_recog_model.pkl",'rb') as f:
        _model = joblib.load(f)
    with open("artifacts\class_dictionary.json") as f:
        _class_name_to_number = json.load(f)
        _class_number_to_name = {v:k for k,v in _class_name_to_number.items()}
    print("succesfully loaded artifacts")

def get_test_img():
    with open("./test_images/b64.txt") as f:
        return f.read()
    
if __name__ == '__main__':
    load_saved_artifacts()
    # print(classify_image(get_test_img(),None))
    # print(classify_image(None, "./test_images/5bd946a6cdbadfc416e59db42422c06c.jpg"))
    print(classify_image(None, "./test_images/messi-CR.jpg"))