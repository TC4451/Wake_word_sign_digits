from picamera2 import Picamera2, Preview
import time

def capture_images(filename1="image1.jpg", filename2="image2.jpg", filename3="image3.jpg", filename4="image4.jpg"):
    """Captures two images with a 5-second delay between them.

    Args:
        filename1 (str, optional): Filename for the first image. Defaults to "phone_image1.jpg".
        filename2 (str, optional): Filename for the second image. Defaults to "phone_image2.jpg".
    """
    try:
        picam2 = Picamera2()

        # Configure the camera (adjust as needed)
        config = picam2.create_preview_configuration(main={"size": (640, 480)}) # Example resolution
        picam2.configure(config)

        picam2.start_preview(Preview.QTGL)  # Start preview
        time.sleep(2)  # Give time to adjust

        picam2.start()  # Start the camera stream

        print("Capturing first image in 2 seconds...")
        time.sleep(2)  # Wait 5 seconds
        picam2.capture_file(filename1)
        print(f"First image saved as {filename1}")

        print("Capturing second image in 2 seconds...")
        time.sleep(2)  # Wait 5 seconds
        picam2.capture_file(filename2)
        print(f"Second image saved as {filename2}")
        
        print("Capturing third image in 2 seconds...")
        time.sleep(2)  # Wait 5 seconds
        picam2.capture_file(filename3)
        print(f"Third image saved as {filename3}")
        
        print("Capturing fourth image in 2 seconds...")
        time.sleep(2)  # Wait 5 seconds
        picam2.capture_file(filename4)
        print(f"Fourth image saved as {filename4}")

        picam2.stop()
        picam2.close()

    except Exception as e:
        print(f"Error: {e}")
