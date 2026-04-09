import cv2
import numpy as np
import os
from picamera2 import Picamera2

# Setup folder
SAVE_DIR = "templates"
os.makedirs(SAVE_DIR, exist_ok=True)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

print("--- FINAL STABLE CAPTURE TOOL ---")
print("1. Aim camera at symbol")
print("2. Ensure the GREEN box covers the WHOLE symbol")
print("3. Press 't' to save | Press 'q' to quit")

try:
    while True:
        frame = picam2.capture_array()
        
        # 1. Vision Pre-processing
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # Using the range you established as working
        color_mask = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([180, 255, 255]))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # FIX: Changed 11,2 to 21,5 to make the QR squares SOLID white instead of outlines
        bin_inv = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 21, 5)
        
        # 2. Clean the mask
        bin_clean = cv2.bitwise_and(bin_inv, color_mask)
        
        # --- QR WELDING LOGIC ---
        # 11x11 is the 'sweet spot' to connect QR squares without losing the box shape
        weld_kernel = np.ones((20, 20), np.uint8) 
        weld_mask = cv2.morphologyEx(bin_clean, cv2.MORPH_CLOSE, weld_kernel)
        # ------------------------
        
        # 3. Find the symbol using the WELDED mask
        contours, _ = cv2.findContours(weld_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        display_frame = frame.copy()
        tight_crop = None 

        if contours:
            c = max(contours, key=cv2.contourArea)
            
            # FIX: Lowered area threshold to 800 to ensure it detects even in varied light
            if cv2.contourArea(c) > 600: 
                x, y, w, h = cv2.boundingRect(c)
                
                # ADD PADDING (20 pixels) - Important for Hu Moments math
                pad = 20
                y1, y2 = max(0, y-pad), min(frame.shape[0], y+h+pad)
                x1, x2 = max(0, x-pad), min(frame.shape[1], x+w+pad)
                
                # Visual confirmation
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # IMPORTANT: Save the detailed mask, not the welded blob
                tight_crop = bin_clean[y1:y2, x1:x2]

        # Show windows
        cv2.imshow("Original (Check Bounding Box)", display_frame)
        cv2.imshow("Mask (Detailed - This is saved)", bin_clean)
        cv2.imshow("Welded (Internal Logic Only)", weld_mask)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('t'):
            if tight_crop is not None:
                name = input("Enter symbol name (e.g. qr_stop): ")
                filename = os.path.join(SAVE_DIR, f"{name}.png")
                cv2.imwrite(filename, tight_crop)
                print(f"--- SUCCESS: Saved template to {filename} ---")
            else:
                print("--- ERROR: No stable symbol detected! ---")
                
        elif key == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
