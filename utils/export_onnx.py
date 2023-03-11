from ultralytics import YOLO


def main():
    model = YOLO("../weights/yolov8n.pt")
    success = model.export(format="onnx")
    print(f"Export success : {success}")


if __name__ == "__main__":
    main()