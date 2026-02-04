from ultralytics import YOLO

model = YOLO('post_training_model/best(1).pt')

model.predict(
    source=r"D:\FOOTBALL_ANALYSIS\input_videos\08fd33_4.mp4",
    save=True,
    project=r"D:\FOOTBALL_ANALYSIS\runs",
    name="detect",
    exist_ok=True
)