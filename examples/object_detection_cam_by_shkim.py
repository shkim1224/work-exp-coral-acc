import cv2
import numpy as np
import time
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

# 모델 및 레이블 파일 경로
model_path = 'test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
label_path = 'test_data/coco_labels.txt'

# 레이블 파일 로드
# with open(label_path, 'r') as f:
#     labels = {int(line.split()[0]): line.strip().split(maxsplit=1)[1] for line in f.readlines()}


def load_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if lines[0].split()[0].isdigit():
            # 'label_id label_name' 형식의 레이블 파일
            return {int(line.split()[0]): line.strip().split(maxsplit=1)[1] for line in lines}
        else:
            # 레이블 이름만 있는 경우
            return {i: line.strip() for i, line in enumerate(lines)}

labels = load_labels(label_path)

# Edge TPU 인터프리터 로드
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# 카메라 초기화
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리
    input_size = common.input_size(interpreter)
    resized_frame = cv2.resize(frame, input_size)
    common.set_input(interpreter, resized_frame)

    # 객체 검출 수행
    interpreter.invoke()
    objs = detect.get_objects(interpreter, 0.4)  # 신뢰도 임계값을 0.4로 설정

    # 객체를 프레임에 그리기
    for obj in objs:
        bbox = obj.bbox.scale(frame.shape[1] / input_size[0], frame.shape[0] / input_size[1])
        cv2.rectangle(frame, (int(bbox.xmin), int(bbox.ymin)), (int(bbox.xmax), int(bbox.ymax)), (0, 255, 0), 2)
        label = '{}: {:.2f}'.format(labels.get(obj.id, obj.id), obj.score)
        cv2.putText(frame, label, (int(bbox.xmin), int(bbox.ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 프레임을 보여주기
    cv2.imshow('Object Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
