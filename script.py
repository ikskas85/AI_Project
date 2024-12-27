import cv2
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime, timedelta

# Загрузка обученной модели YOLO
model = YOLO(r"PATH_TO_MODEL")

# Настройки видео и модели
# VIDEO_PATH = r"https://media.gov39.ru/webcam-rec/mapp_gzhehodki.stream/playlist.m3u8"
VIDEO_PATH = r"PATH_TO_VIDEO/STREAM"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MIDDLE_X = FRAME_WIDTH // 2  # Вертикальная линия для разделения полос
TARGET_Y = 220  # Линия шлагбаума
FRAME_SKIP = 5  # Пропуск кадров для ускорения обработки
CONF_THRESHOLD = 0.7
IOU_THRESHOLD = 0.3

# Состояние трекинга
tracked_ids = {"left": set(), "right": set()}  # Учет ID объектов по полосам
crossed_ids = set()  # Учет ID объектов, пересёкших шлагбаум
last_check_time = datetime.now()  # Последнее время проверки статистики

# Открытие видеофайла
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue  # Пропуск кадров для ускорения

    # Изменение размера кадра
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # Обработка кадра через YOLO
    results = model.track(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device="cpu", verbose=False)

    # Если нет обнаруженных объектов, пропускаем кадр
    if not results[0].boxes:
        continue

    # Текущая статистика объектов на кадре
    lane_counts = {"left": defaultdict(int), "right": defaultdict(int)}

    # Извлечение координат, классов и ID объектов
    boxes = results[0].boxes.xywh.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    classes = results[0].boxes.cls.int().cpu().tolist()

    for box, track_id, cls in zip(boxes, track_ids, classes):
        x, y, _, _ = box  # Центр объекта (x, y)
        cls_name = model.names[cls]  # Название класса объекта

        # Определение полосы: левая или правая
        in_left_lane = x < MIDDLE_X
        lane = "left" if in_left_lane else "right"
        lane_counts[lane][cls_name] += 1  # Увеличиваем счетчик объектов на текущем кадре

        # Проверка пересечения шлагбаума
        if TARGET_Y - 10 <= y <= TARGET_Y + 10 and track_id not in crossed_ids:
            side = "to_camera" if in_left_lane else "from_camera"
            print("\n=== CROSSING ===")
            print(f"{side} - {datetime.now().strftime('%H:%M:%S')} - {cls_name} - {track_id}")
            crossed_ids.add(track_id)  # Учет объекта, пересекшего шлагбаум

    # Вывод текущей статистики каждые 3 минуты
    if datetime.now() - last_check_time >= timedelta(minutes=1):
        print("\n=== CURRENT STATE ===")
        for lane, counts in lane_counts.items():
            direction = "to_camera" if lane == "left" else "from_camera"
            print(f"Lane: {lane} Direction: {direction}")
            for cls_name, count in counts.items():
                print(f"  Type: {cls_name}, Count: {count}")
        last_check_time = datetime.now()

    # Аннотирование кадра
    annotated_frame = results[0].plot()

    # Отображение разделительной линии и линии шлагбаума
    cv2.line(annotated_frame, (MIDDLE_X, 0), (MIDDLE_X, FRAME_HEIGHT), (0, 255, 0), 2)  # Вертикальная линия
    cv2.line(annotated_frame, (0, TARGET_Y), (FRAME_WIDTH, TARGET_Y), (0, 0, 255), 3)  # Линия шлагбаума

    #Показ аннотированного кадра
    cv2.imshow("Car Tracker", annotated_frame)
    # cv2.imshow("Car Tracker", frame)

    # Завершение работы по нажатию клавиши "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
