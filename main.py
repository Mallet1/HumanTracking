# import cv2
# from tracker import *
#
# # Create tracker object
# tracker = EuclideanDistTracker()
#
# cap = cv2.VideoCapture("highway.mp4")
# capture = cv2.VideoCapture(0)
#
# # Object detection from Stable camera
# object_detector = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=1000)
#
# while True:
#     ret, frame = cap.read()
#     height, width, _ = frame.shape
#
#     # Extract Region of interest
#     # roi = frame
#
#     isTrue, roi = capture.read()
#
#     # 1. Object Detection
#     mask = object_detector.apply(roi)
#     _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     detections = []
#     for cnt in contours:
#         # Calculate area and remove small elements
#         area = cv2.contourArea(cnt)
#         if area > 1000:
#             cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
#             x, y, w, h = cv2.boundingRect(cnt)
#
#
#             detections.append([x, y, w, h])
#
#     # 2. Object Tracking
#     boxes_ids = tracker.update(detections)
#     for box_id in boxes_ids:
#         x, y, w, h, id = box_id
#         cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
#         cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
#
#     cv2.imshow("roi", roi)
#     # cv2.imshow("Frame", frame)
#     cv2.imshow("Mask", mask)
#
#     key = cv2.waitKey(30)
#     if key == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()





# import the necessary packages
# import numpy as np
# import cv2
#
# # initialize the HOG descriptor/person detector
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#
# cv2.startWindowThread()
#
# # open webcam video stream
# cap = cv2.VideoCapture(0)
#
# # the output will be written to output.avi
# out = cv2.VideoWriter(
#     'output.avi',
#     cv2.VideoWriter_fourcc(*'MJPG'),
#     15.,
#     (640, 480))
#
# while (True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # resizing for faster detection
#     frame = cv2.resize(frame, (640, 480))
#     # using a greyscale picture, also for faster detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#
#     # detect people in the image
#     # returns the bounding boxes for the detected objects
#     boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
#
#     boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
#
#     for (xA, yA, xB, yB) in boxes:
#         # display the detected boxes in the colour picture
#         cv2.rectangle(frame, (xA, yA), (xB, yB),
#                       (0, 255, 0), 2)
#
#     # Write the output video
#     out.write(frame.astype('uint8'))
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#     cv2.imshow('gray', gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# # and release the output
# out.release()
# # finally, close the window
# cv2.destroyAllWindows()
# cv2.waitKey(1)


import cv2
import imutils

# Initializing the HOG person
# detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Reading the video stream
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image,
                               width=min(400, image.shape[1]))

        # Detecting all the regions
        # in the Image that has a
        # pedestrians inside it
        (regions, _) = hog.detectMultiScale(image,
                                            winStride=(4, 4),
                                            padding=(4, 4),
                                            scale=1.05)

        # Drawing the regions in the
        # Image
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y),
                          (x + w, y + h),
                          (0, 0, 255), 2)

        # Showing the output Image
        cv2.imshow("Image", image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()