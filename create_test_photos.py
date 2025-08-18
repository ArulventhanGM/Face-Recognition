#!/usr/bin/env python3

import cv2
import numpy as np
import os

# Create a simple test image
test_img = np.ones((200, 200, 3), dtype=np.uint8) * 240
cv2.putText(test_img, 'Test Student', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.putText(test_img, 'Photo', (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# Save to uploads folder
os.makedirs('uploads', exist_ok=True)
cv2.imwrite('uploads/test_student_photo.jpg', test_img)
print('Test photo created: uploads/test_student_photo.jpg')

# Also create one specifically for student ID 1234567890
cv2.imwrite('uploads/1234567890.jpg', test_img)
print('Test photo created: uploads/1234567890.jpg')

# Create for dharan123
cv2.imwrite('uploads/dharan123.jpg', test_img)
print('Test photo created: uploads/dharan123.jpg')
