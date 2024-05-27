from ultralytics import YOLO
import cv2

model = YOLO("yolov8x-pose.pt")

path = "ex1.jpg"
results = model(path, save=True, save_txt=True, save_conf=True)
img = cv2.imread(path)

keypoints = results[0].keypoints.data

print(keypoints)


skeleton  = [(16,14),(14,12),(15,13),(13,11),(6,12),(5,11),(11,12),
			(6,8),(7,9),(8,10),(9,7),(5,6)]

face_keypoint = [0,1,2,3,4]

start_point = (int(1280 * 0.498858), int(720 * 0.302394))
end_point = (int(1280 * 0.505837), int(720 * 0.295608))

img = cv2.line(img,start_point,end_point, (0,0, 255),3)
cv2.circle()



cv2.imshow('imshow_test', img)
cv2.waitKey(0) #待機時間、ミリ秒指定、0の場合はボタンが押されるまで待機
cv2.destroyAllWindows()