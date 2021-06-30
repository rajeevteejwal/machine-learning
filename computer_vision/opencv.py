import cv2 as cv
import numpy as np
import face_recognition_models as frm


# frame = cv.imread('D:/Repos_ML/images/cat.jpg')
# cv.imshow("cat", frame)
# video = cv.VideoCapture('D:/Repos_ML/images/sample_1920x1080.mp4')
'''Video capturing '''
'''video = cv.VideoCapture(0)
video.set(3,1920)
video.set(4,720)
video.set(10,1000)
while True:
    success, img = video.read()
    frame = cv.flip(img, flipCode=1)
    cv.imshow("My video", frame)
    if cv.waitKey(1) == ord('q'):
        break

video.release()
cv.destroyAllWindows()
'''
# common functions in open cv
'''img = cv.imread('D:/Repos_ML/images/cat.jpg')
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray, (5, 5), 0)
imgCanny = cv.Canny(imgGray, 100, 100)
dilated_kernal = np.ones((5, 5), dtype=np.uint8)
imgDilated = cv.dilate(imgCanny, dilated_kernal, iterations=1)
imgEroded = cv.erode(imgDilated, dilated_kernal, iterations=1)

cv.imshow("Gray image", imgGray)
cv.imshow("Blur image", imgBlur)
cv.imshow("Canny Image", imgCanny)
cv.imshow("Dialeted Image", imgDilated)
cv.imshow("Eroded Image", imgEroded)'''

# Resizing and cropping
'''img = cv.imread('D:/Repos_ML/images/cat.jpg')
print(img.shape)
cv.imshow("Image", img)
imgResize = cv.resize(img, (250, 300))
cv.imshow("Resize Image", imgResize)

imgCropped = img[0:200, 200:500]
cv.imshow("Resize Image", imgCropped)'''

# Image translation
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)
#rotate
def rotate(img, angle, rotPoint = None):
    if rotPoint == None :
        rotPoint = (img.shape[1]//2, img.shape[0]//2)
    rotateMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (img.shape[1],img.shape[0])
    return cv.warpAffine(img, rotateMat, dimensions)

# Drawing texts and shapes on image
img = cv.imread('D:/Repos_ML/images/cat.jpg')
'''img = np.random.randint(low=0, high=256, size=(500, 500, 3))
img = np.zeros((500, 500, 3))'''
'''imgTransed = translate(img,50,150)
cv.imshow("Image", imgTransed)

imgRotated = rotate(img, -90,(150,250))
imgRotatedTranslated = translate(imgRotated,150,-250)
cv.imshow("Rotated ", imgRotated)
cv.imshow("Rotated Translated", imgRotatedTranslated)
'''
# flip

'''fliped = cv.flip(img,1)
cv.imshow("Fliped", fliped)'''
###
# contour
#image = cv.imread('D:/Repos_ML/images/cat.jpg')
#gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#canny = cv.Canny(image, 125, 175)

#cv.imshow("Gray", canny)

#contours, hierarchies = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

#for contour in contours:
#    print(cv.contourArea(contour))

#print(len(contours))


def doNothing(x):
    pass

# masking
'''cv.namedWindow('Trackbar')

cv.createTrackbar("L-H", "Trackbar", 0, 179, doNothing)
cv.createTrackbar("L-S", "Trackbar", 0, 255, doNothing)
cv.createTrackbar("L-V", "Trackbar", 0, 255, doNothing)
cv.createTrackbar("U-H", "Trackbar", 179, 179, doNothing)
cv.createTrackbar("U-S", "Trackbar", 255, 255, doNothing)
cv.createTrackbar("U-V", "Trackbar", 255, 255, doNothing)

video_frame = cv.VideoCapture(0)
while True:
    _, frame = video_frame.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    l_h = cv.getTrackbarPos("L-H", "Trackbar")
    l_s = cv.getTrackbarPos("L-S", "Trackbar")
    l_v = cv.getTrackbarPos("L-V", "Trackbar")
    u_h = cv.getTrackbarPos("U-H", "Trackbar")
    u_s = cv.getTrackbarPos("U-S", "Trackbar")
    u_v = cv.getTrackbarPos("U-V", "Trackbar")
    lower_limit = np.array([l_h, l_s, l_v])
    upper_limit = np.array([u_h, u_s, u_v])
    mask = cv.inRange(hsv, lower_limit, upper_limit)
    cv.imshow("Mask frame",mask)
    cv.imshow("Masking example", frame)
    # Bitwise-AND between mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow("Result image", res)'''

# Date 30 March 2021
'''vf = cv.VideoCapture(0)

while True:
    ret, frame = vf.read()
    cv.imshow("Original", frame)
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray Image", gray_image)
    blur_image = cv.GaussianBlur(gray_image, (3, 3), 1)
    cv.imshow("Blur Image", blur_image)
    canny = cv.Canny(frame, 50, 200)
    cv.imshow("Edge Image", canny)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()'''
#cv.waitKey(0)


'''face recognition code '''


def image_encodings():
    image = frm .load_image_file("computer_vision/datasets/train/Rajeev Teejwal/1.jpg")
    face_locations = frm.face_locations(image)


image_encodings()
