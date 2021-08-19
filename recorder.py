import cv2

capture1 = cv2.VideoCapture(0)
capture2 = cv2.VideoCapture(1)
capture3 = cv2.VideoCapture(3)

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
videoWriter1 = cv2.VideoWriter('video1.avi', fourcc, 30.0, (128, 128))
videoWriter2 = cv2.VideoWriter('video1.avi', fourcc, 30.0, (128, 128))
videoWriter3 = cv2.VideoWriter('video1.avi', fourcc, 30.0, (128, 128))

while (True):

    ret1, frame1 = capture1.read()
    ret2, frame2 = capture2.read()
    ret3, frame3 = capture3.read()

    if ret1 and ret2 and ret3:
        cv2.imshow('video1', frame1)
        cv2.imshow('video2', frame2)
        cv2.imshow('video3', frame3)
        videoWriter1.write(frame1)
        videoWriter2.write(frame2)
        videoWriter3.write(frame3)

    if cv2.waitKey(1) == 27:
        break

capture1.release()
capture2.release()
capture3.release()
videoWriter1.release()
videoWriter2.release()
videoWriter3.release()

cv2.destroyAllWindows()