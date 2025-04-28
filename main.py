'''Person counter using YOLO
Code by: Mohammed Abdullah Ba'ashar
Website: https://www.bytsnbytes.com
Github: https://github.com/mabaashar
'''
#--------------------------
import ultralytics
from ultralytics import YOLO
import cv2
from ultralytics import solutions

#read the video feed
cap = cv2.VideoCapture("path/to/input")
#for webcam use 0 as the source
#cap = cv2.VideoCapture(0)

#check if video is valid
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))

#output video after processing
video_writer = cv2.VideoWriter("path/to/output",
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (w, h))

# Define region points to scan
# region_points = [(20, 400), (1080, 400)]  # For line counting
#region_points = [(0, 0), (, 400), (1080, 360), (20, 360)]  # For rectangle region counting
region_points = [(0, 0), (w, 0), (w, h), (0, h)] #for full screen
# region_points = [(20, 400), (1080, 400, (1080, 360), (20, 360), (20, 400)]  # For polygon region counting

# Init ObjectCounter
counter = solutions.ObjectCounter(
    show=True,  # Display the output
    region=region_points,  # Pass region points
    model="yolo11n.pt",
    classes=[0],  # count people only with COCO pretrained model.
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=2,  # Adjust the line width for bounding boxes and text display
)

#define total count
total_count = 0

# Process video
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    results = counter(im0)  # count the objects
    #print the total count of persons detected
    cv2.putText(im0,str("Total number of people: ")+str(results.total_tracks), (0,200), 0, 1, [225, 0, 0], thickness=2, lineType=cv2.LINE_AA)
    video_writer.write(results.plot_im)   # write the video frames

cap.release()   # Release the capture
video_writer.release()
cv2.destroyAllWindows()