import cv2
import os

faceCascade = cv2.CascadeClassifier("dataset/haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("dataset/haarcascade_smile.xml")

if faceCascade.empty() or smileCascade.empty():
    raise IOError("Could not load one or more cascade files. Please check the 'dataset' folder.")

video = cv2.VideoCapture(0)

cnt = 1

save_dir = './saved_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

while True:
    success, img = video.read()
    if not success:
        break

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayImg, 1.1, 4)

    keyPressed = cv2.waitKey(1)

    smiling = False
    saved_this_frame = False

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        roi_gray = grayImg[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        smiles = smileCascade.detectMultiScale(roi_gray, 1.8, 15)

        if len(smiles) > 0:
            smiling = True

        for (sx, sy, sw, sh) in smiles:
            if sy > h // 2:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 3)

        if len(smiles) > 0 and not saved_this_frame and cnt <= 3:
            abs_save_dir = os.path.abspath(save_dir)
            path = os.path.join(abs_save_dir, f"{cnt}.jpg")
            success = cv2.imwrite(path, img)
            cnt += 1
            saved_this_frame = True

    if cnt > 3:
        video.release()
        cv2.destroyAllWindows()
        break

    if smiling:
        cv2.putText(img, "Smiling", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('live video', img)
    if keyPressed & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
