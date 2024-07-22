import cv2
import numpy as np

main_image_path = 'C:/Users/musta/OneDrive/Desktop/ImageObjectDetector/main.jpg'
template_image_path = 'C:/Users/musta/OneDrive/Desktop/ImageObjectDetector/small.png'

main_image = cv2.imread(main_image_path)
template = cv2.imread(template_image_path)

if main_image is None:
    print(f"Error: Could not load main image at {main_image}")
    exit()
if template is None:
    print(f"Error: Could not load template image at {template}")
    exit()
    
    
main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template_gray.shape[::-1]

result = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)

threshold = 0.8
loc = np.where(result >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(main_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

cv2.imshow('Detected Football', main_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
