import cv2
import sys

# get the path of image and xml file
image = sys.argv[1]
casc = sys.argv[2]

# create a cascade
face_cascade = cv2.CascadeClassifier(casc)

# get color_image & corresponding gray image
color_image = cv2.imread(image)
gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# detect the faces in the image
# variable faces contains all the detected faces
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=5,minSize=(25,25))

print(f"Algo found total {len(faces)} faces")

# draw red rectange around faces
for (x,y,w,h) in faces:
     cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow("Faces found", color_image)
cv2.waitKey(0)