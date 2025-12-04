import cv2
import dlib
import logging
import numpy as np
from imutils import face_utils
from skimage import transform as trans
import argparse
import os
import sys
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)

def create_logger(log_path):
    # Create logger object
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler and set the formatter
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(fh)

    # Add a stream handler to print to console
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts

def extract_aligned_face_dlib(face_detector, predictor, image, res=256, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img, None

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face, mask_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        if len(face_align) == 0:
            return None, None, None
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmark, mask_face
    else:
        return None, None, None

def facecrop(video_path):
        face_detector = dlib.get_frontal_face_detector()
        predictor_path = f'{base_path}/shape_predictor_81_face_landmarks.dat'

        face_predictor = dlib.shape_predictor(predictor_path)
    
        # Open the video file
        cap_org = cv2.VideoCapture(str(video_path))
 
        # Get the number of frames in the video
        frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
        # Get the mode
        frame_idxs = np.linspace(0, frame_count_org - 1, 32, endpoint=True, dtype=int)
  
        face_frames = []
        face_crops = []
        img_frames = []
        # Iterate through the frames
        for cnt_frame in range(frame_count_org):
            _, frame_org = cap_org.read()

            # Check if the frame is one of the frames to extract
            if cnt_frame not in frame_idxs:
                continue

            if frame_org is not None:
                cropped_face, _, _ = extract_aligned_face_dlib(face_detector, face_predictor, frame_org)
            else:
                continue

            if cropped_face is not None:
                face_crops.append(cropped_face)
                face_frames.append(frame_org)
            else:
                frame_org = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
                img_frames.append(frame_org)
                continue
            
        # Release the video capture
        cap_org.release()

        return face_crops, face_frames, img_frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-r', type=str, default='//', help='')
    args = parser.parse_args()

    frames = facecrop(args.video_path)

    print(len(frames))

    
