import os
import numpy as np
import cv2

# ------------------------------
# ENTER YOUR PARAMETERS HERE:
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
LENGTH_PX = 640   # total length of the page in pixels
MARGIN_PX = 20    # size of the margin in pixels
SAVE_NAME = 'ChArUco_Marker.png'
# ------------------------------

def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
    cv2.imshow("img", img)
    cv2.waitKey(2000)
    cv2.imwrite(SAVE_NAME, img)

create_and_save_new_board()

# ------------------------------
# ENTER YOUR REQUIREMENTS HERE:
ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
# ...
PATH_TO_YOUR_IMAGES = '/Users/Ed/Downloads/Calibration_Images'
# ------------------------------

def calibrate_and_save_parameters():
    # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    # Load PNG images from folder
    image_files = [os.path.join(PATH_TO_YOUR_IMAGES, f) for f in os.listdir(PATH_TO_YOUR_IMAGES) if f.endswith(".png")]
    image_files.sort()  # Ensure files are in order

    all_charuco_corners = []
    all_charuco_ids = []

    for image_file in image_files:
        image = cv2.imread(image_file)
        image_copy = image.copy()
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
        
        # If at least one marker is detected
        if len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
            if charuco_retval:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)


    # PenguinPi Charuco calibration script
    import numpy as np
    import cv2
    import os
    import glob

    # Charuco board parameters
    CHARUCO_ROWS = 7
    CHARUCO_COLS = 5
    SQUARE_LENGTH = 0.04  # meters
    MARKER_LENGTH = 0.02  # meters

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    charuco_board = cv2.aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        aruco_dict
    )

    # Load all calibration images taken by PenguinPi
    image_dir = 'images'
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    print(f"Found {len(image_paths)} images in '{image_dir}' for calibration.")

    all_corners = []
    all_ids = []
    image_size = None

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to load image: {img_path}")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if len(corners) > 0:
            resp, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=charuco_board
            )
            if resp > 10:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                image_size = gray.shape[::-1]
                print(f"Accepted image: {img_path} ({resp} corners)")
            else:
                print(f"Not enough Charuco corners in {img_path} ({resp})")
        else:
            print(f"No ArUco markers detected in {img_path}")

    if len(all_corners) > 0:
        print("Calibrating...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=charuco_board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None
        )
        print("Calibration RMS error:", ret)
        print("Camera matrix:\n", camera_matrix)
        print("Distortion coefficients:\n", dist_coeffs)
        # Ensure output directory exists
        out_dir = os.path.join('calibration', 'param')
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(os.path.join(out_dir, 'intrinsic.txt'), camera_matrix, delimiter=',', fmt='%.18e')
        np.savetxt(os.path.join(out_dir, 'distortion.txt'), dist_coeffs, delimiter=',', fmt='%.18e')
        print(f"Camera matrix saved to {os.path.join(out_dir, 'intrinsic.txt')}")
        print(f"Distortion coefficients saved to {os.path.join(out_dir, 'distortion.txt')}")
    else:
        print("Not enough valid images for calibration.")