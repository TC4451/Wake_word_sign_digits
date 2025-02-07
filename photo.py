import cv2

def capture_and_resize(width=64, height=64):
    """Captures an image from the webcam and resizes it."""

    cap = cv2.VideoCapture(0)  # 0 usually refers to the default webcam

    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

    ret, frame = cap.read() # Reads one frame from the camera
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

    # Resize the frame
    resized_frame = cv2.resize(frame, (width, height))

    cap.release() # Release the capture

    return resized_frame

def display_image(image):
    """Displays an image using OpenCV."""
    cv2.imshow('Captured and Resized Image', image)
    cv2.waitKey(0)  # Waits for a key press to close the window
    cv2.destroyAllWindows()

def save_image(image, filename="captured_image.jpg"):
    """Saves an image to a file."""
    cv2.imwrite(filename, image)
    print(f"Image saved to {filename}")

# Example usage:
width = 100
height = 100
resized_image = capture_and_resize(width, height)

display_image(resized_image)  # Display the captured and resized image
save_image(resized_image, "captured_100x100.jpg") # Save the image as captured_64x64.jpg

# If you want to convert to RGB from BGR (OpenCV's default color format):
rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
save_image(rgb_image, "captured_64x64_rgb.jpg") # Save the image as captured_64x64_rgb.jpg