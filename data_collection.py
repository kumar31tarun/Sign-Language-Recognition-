import cv2
import os
import numpy as np

# Create a list of symbols to collect data for dataset
symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y','0','1','2','3','4','5','6','7','8','9']

# Create a directory to store the data
#data_dir is a variable that stores the absolute path of the current directory through abspath which it gets from dirname(__file__)
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

#setting up the frame of the video with dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
# Set the dimensions of the captured frame

# Initialize the current symbol to 'A'
current_symbol = 'A'

# Create a directory for the current symbol
current_symbol_dir = os.path.join(data_dir, current_symbol)
if not os.path.exists(current_symbol_dir):
    os.makedirs(current_symbol_dir)

# Set a flag to indicate when to stop collecting data
stop_collecting = False

# Start collecting data until the '/' key is pressed
image_count=0
#setting up a previous key to check if the new key is pressed
prev_key=None

#running the loop until the stop collecting becomes boolean True
while not stop_collecting:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    #setting up the Rectangular frame on the frame with width being 5
    cv2.rectangle(frame,(40,40),(300,300),(0,255,0),5)

    #cv2.putText(frame,'test',(20,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the frame
    cv2.imshow('ROI', gray[40:300,40:300])
    #Resizing the image to 28x28 format
    gray = cv2.resize(gray[40:300,40:300], (28, 28), interpolation = cv2. INTER_AREA)
    
    #showing the gray frame with 28x28 pixels
    cv2.imshow('sclaed and gray',gray)
    #printing on the 'frame' what symbol is being collected and how much is collected 
    cv2.putText(frame,'collecting image for : {}'.format(current_symbol),(20,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    cv2.putText(frame,str(image_count),(20,400),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)

    #To show the frame with BGR colour
    cv2.imshow('Frame',frame)
    
    key = cv2.waitKey(1) & 0xFF 
    # Check if the user pressed the '/' key to stop collecting data
    if key == ord('/'):
        stop_collecting = True

    # Check if the user pressed a key corresponding to an symbol
    if key in [ord(s) for s in symbols]:
        #if it did it sets that key to the current_symbol
            if current_symbol!=chr(key):
                current_symbol = chr(key)
                image_count=0
                
            
            
            image_count+=1
            
            # Update the current symbol and create a directory for it if it doesn't exist
            current_symbol_dir = os.path.join(data_dir, current_symbol)
            if not os.path.exists(current_symbol_dir):
                os.makedirs(current_symbol_dir)

            # Generate a filename for the current frame
            filename = os.path.join(current_symbol_dir, f"{current_symbol}_{len(os.listdir(current_symbol_dir)) + 1}.jpg")


           # Save the current frame to disk
            cv2.imwrite(filename, gray)

    
# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
