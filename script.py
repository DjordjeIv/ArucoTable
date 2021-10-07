import numpy as np
import cv2 as cv
import glob
from cv2 import aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
markerLength = 40
markerSeparation = 8
aruco_board = aruco.GridBoard_create(5, 7, markerLength, markerSeparation, aruco_dict)
aruco_params = aruco.DetectorParameters_create()

counter, corner_list, id_list = [], [], []
images = glob.glob('calib_image_*.jpg')
height = 0
width = 0
first = True
for fname in images:
    image = cv.imread(fname)
    height, width = image.shape[:2]
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(image_gray, aruco_dict, parameters=aruco_params)
    if first == True:
        corner_list = corners
        id_list = ids
        first = False
    else:
        corner_list = np.vstack((corner_list, corners))
        id_list = np.vstack((id_list, ids))
    counter.append(len(ids))

counter = np.array(counter)

ret, matrix, dist_coeff, rvecs, tvecs = aruco.calibrateCameraAruco(corner_list, id_list, counter, aruco_board, (width, height), None, None)


cap = cv.VideoCapture("Aruco_board.mp4")
i = 0
while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    newCameraMatrix, validPixROI = cv.getOptimalNewCameraMatrix(matrix, dist_coeff, (w, h), 1, (w, h))
    frame = cv.undistort(frame, matrix, dist_coeff, None, newCameraMatrix)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=aruco_params)
    aruco.refineDetectedMarkers(frame_gray, aruco_board, corners, ids, rejectedImgPoints)
    output = aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
    retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, aruco_board, matrix, dist_coeff, None, None)
    if retval != 0:
        output = aruco.drawAxis(output, matrix, dist_coeff, rvec, tvec, 128)

    cv.imshow("OUTPUT", output)
    cv.imwrite("output_" + str(i) + ".png", output)
    i += 1
    cv.waitKey(1)

cap.release()
cv.destroyAllWindows()
