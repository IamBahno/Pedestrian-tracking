from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")


    # Train the model
    results = model.train(
        data="data.yaml",  # Path to the dataset config file
        epochs=60,                 # Number of training epochs
        imgsz=640,                 # Image size (resize to 640x640)
        batch=64,                  # '-1' Batch size (adjust for your GPU/CPU)
        val=True,                  # Enable validation
        device=0,
        workers=4,
        seed=42,
        patience=10
    )

    # it is save automatically

if __name__ == "__main__":
    main()

