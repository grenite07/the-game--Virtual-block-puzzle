import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

# --- 1. SETUP ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)

# --- 2. THE DRAGGABLE RECTANGLE CLASS ---
class DragRect():
    def __init__(self, posCenter, size=[100, 100], color=(255, 0, 255)):
        self.posCenter = posCenter # [x, y]
        self.size = size           # [width, height]
        self.color = color         # Different color for each box

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # Check if the fingertip (cursor) is inside this rectangle
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
           cy - h // 2 < cursor[1] < cy + h // 2:
            
            # If inside, update the position to match the finger (DRAG)
            self.posCenter = cursor
            self.color = (0, 255, 0) # Turn Green when holding
        else:
            # Revert color when not holding (Note: simplistic logic)
             if self.color == (0, 255, 0): 
                 self.color = (255, 0, 255) # This will reset color (simple version)

# --- 3. CREATE BLOCKS ---
rectList = []
# Create 5 blocks at different positions
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

# --- 4. MAIN LOOP ---
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    # Detect Hand
    hands, img = detector.findHands(img, flipType=False)
    
    cursor = None # Variable to store finger tip position

    if hands:
        lmList = hands[0]['lmList']
        # Point 8 is Index Tip, Point 12 is Middle Finger Tip (for stability)
        # But we will use Index (8) and Thumb (4) for Pinching
        x8, y8 = lmList[8][0], lmList[8][1]
        x4, y4 = lmList[4][0], lmList[4][1]
        
        # Calculate Distance (The Pinch)
        length, info, img = detector.findDistance((x8, y8), (x4, y4), img)
        
        # IF PINCHED (Distance < 40), we set the cursor position
        if length < 40:
            cursor = [x8, y8] # The "Mouse" is active

    # --- 5. UPDATE AND DRAW BLOCKS ---
    
    # Make a transparent layer for cool effect
    imgNew = np.zeros_like(img, np.uint8)

    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        
        # If we are pinching (cursor exists), try to update the rect position
        if cursor:
            rect.update(cursor)
        
        # Draw the rectangle
        # Using simple math to find Top-Left corner from Center
        x1 = cx - w // 2
        y1 = cy - h // 2
        
        # Draw on the transparent layer
        cv2.rectangle(imgNew, (x1, y1), (x1 + w, y1 + h), rect.color, cv2.FILLED)
        cvzone.cornerRect(imgNew, (x1, y1, w, h), 20, rt=0) # Add cool corners

    # Merge the transparent layer with the original image
    out = img.copy()
    alpha = 0.5 # Transparency level
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Virtual Drag and Drop", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break