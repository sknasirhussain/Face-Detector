import cv2
face_cap = cv2.CascadeClassifier("C:/Users/SK NASIR HUSSAIN/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

#opening the camera
video_cap = cv2.VideoCapture(0)


while True:
    ret, video_data = video_cap.read()

    #converting the image to a B/W one because facial features can be understood better in grayscale
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    #detecting face in the rectangle box
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    #creating the box
    for (x,y,w,h) in faces:
        cv2.rectangle(video_data, (x,y), (x+w,y+h), (0,255,0), 1)
    
    #displaying the video data
    cv2.imshow("video_live", video_data)

    #to clode the window press "a" on the keyboard after waiting for 10 ms
    if cv2.waitKey(10) == ord("a") :
        break
video_cap.release()
