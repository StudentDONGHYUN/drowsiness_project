import mediapipe
from mediapipe.tasks.python import vision

print('mediapipe version:', mediapipe.__version__)
print('FaceLandmarkerOptions.__init__ signature:')
print(vision.FaceLandmarkerOptions.__init__.__doc__)

print('\nFaceLandmarkerOptions dir:')
print(dir(vision.FaceLandmarkerOptions))

try:
    options = vision.FaceLandmarkerOptions(
        base_options=None,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        interpolate_landmarks=True
    )
    print('\ninterpolate_landmarks 옵션 정상 적용됨!')
except TypeError as e:
    print('\nTypeError 발생:', e)
except Exception as e:
    print('\n기타 예외 발생:', e) 