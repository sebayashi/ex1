import cv2
from ultralytics import YOLO

# YOLOモデルのロード
model = YOLO("yolov8x-pose.pt")

# 画像をモデルに通して結果を取得
results = model("ex1.jpg", save=True, save_txt=True, save_conf=True)
keypoints = results[0].keypoints

# キーポイントの取得
points = keypoints.data[0]

# スケルトン定義
skeleton = [(16,14),(14,12),(15,13),(13,11),(6,12),(5,11),(11,12),
            (6,8),(7,9),(8,10),(9,7),(5,6),(5,7)]

# 画像の読み込み
image = cv2.imread("ex1.jpg")

# キーポイントを結ぶ直線とその継ぎ目に円を描画
for joint in skeleton:
    pt1 = [int(points[joint[0]][0]), int(points[joint[0]][1])]
    pt2 = [int(points[joint[1]][0]), int(points[joint[1]][1])]
    
    cv2.line(image, tuple(pt1), tuple(pt2), (0, 0, 255), 2)  # 黄色で直線を描画
    cv2.circle(image, tuple(pt1), 5, (0, 255, 255), -1)  # 直線の継ぎ目に円を描画
    cv2.circle(image, tuple(pt2), 5, (0, 255, 255), -1)  # 直線の継ぎ目に円を描画

# 結果の画像をJPGとして保存
output_filename = "output.jpg"
cv2.imwrite(output_filename, image)
print(f"画像を保存しました: {output_filename}")
