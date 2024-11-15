from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8x.pt')

def train_model():
    # model = YOLO('C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\MulitpleObjectTracking\\models\\runs\\detect\\train14\\weights\\last.pt')
    # model = YOLO('C:\\Users\\aimlc\\OneDrive\\Desktop\\Sowmesh\\MulitpleObjectTracking\\models\\runs\\detect\\train11\\weights\\last.pt')
    model = YOLO("yolov8x.pt") 
    # Train the model
    results = model.train(data="D:\\Sowmesh\\licence_plate_rec_dataset\\license_plate_rec\\data.yaml", epochs=10, batch=32, imgsz=640,workers = 16)

if __name__ == "__main__":

    train_model()