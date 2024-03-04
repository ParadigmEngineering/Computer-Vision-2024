import cv2
import numpy as np

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        if len(line) > 0:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[1][0]
            y2 = line[1][1]
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lines2(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        if len(line) > 0:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def weighted_img(img, initial_img, α=0.8, β=1., λ=0):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def hough_lines(img, rho=2, theta=np.pi/180, threshold=20, min_line_len=25, max_line_gap=10):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def average_lines(lines):
    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            gradient = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0  # Avoid division by zero
            intercept = y1 - gradient * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)  # Pythagorean Theorem

            # Separate lines into left and right based on gradient
            if gradient < 0:
                left_lines.append((gradient, intercept, length))
            else:
                right_lines.append((gradient, intercept, length))

    # Average the gradient and intercept, weighted by line length
    left_lane = np.average(left_lines, axis=0, weights=[line[2]**3 for line in left_lines]) if len(left_lines) > 0 else None
    right_lane = np.average(right_lines, axis=0, weights=[line[2]**3 for line in right_lines]) if len(right_lines) > 0 else None

    return left_lane, right_lane

def process_frame(frame):
    # Convert the frame to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(grayscale_frame, (5, 5), 0)

    # Use Canny edge detector to find edges in the image
    edges = cv2.Canny(blurred, 50, 150)

    return edges

def region_of_interest(image, vertices):
    # Create a blank mask
    mask = np.zeros_like(image)

    # Defining a 3-channel or 1-channel color to fill the mask with
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def recommend_direction(lane_lines):
    print(lane_lines)
    if (len(lane_lines) == 2):
        gradients = []
        for line in lane_lines:
                x1 = line[0][0]
                y1 = line[0][1]
                x2 = line[1][0]
                y2 = line[1][1]
                gradient = (y1 - y2) / (x2 - x1) if (x2 - x1) != 0 else 0  # Avoid division by zero
                gradients.append(gradient)

        mean_gradient = (gradients[0] + gradients[1]) /2
        
        #Threshold Difference (Modify to change the sensitivity of the system to turns)
        threshold = 0.75
        threshold_light = 0.2
        print(gradients)
        print(mean_gradient)
        if (mean_gradient < -threshold):
            return "Medium Right"
        elif (mean_gradient > threshold):
            return "Medium Left"
        elif (mean_gradient < -threshold_light):
            return "Slight Right"
        elif (mean_gradient > threshold_light):
            return "Slight Left"
        else:
            return "Straight"


# Image Paths
turn_1 = "C:/Paradigm/Paradigm_CV_2024/Driving Pictures/turn 1.jpg"
turn_2 = "C:/Paradigm/Paradigm_CV_2024/Driving Pictures/turn 2.jpg"
turn_3 = "C:/Paradigm/Paradigm_CV_2024/Driving Pictures/turn 3.jpg"
turn_4 = "C:/Paradigm/Paradigm_CV_2024/Driving Pictures/turn 4.jpg"
turn_5 = "C:/Paradigm/Paradigm_CV_2024/Driving Pictures/turn 5.jpg"
turn_6 = "C:/Paradigm/Paradigm_CV_2024/Driving Pictures/turn 6.jpg"
straight_2 = "C:/Paradigm/Paradigm_CV_2024/Driving Pictures/straight 2.jpg"
straight_3 = "C:/Paradigm/Paradigm_CV_2024/Driving Pictures/straight 3.jpg"
straight_4 = "C:/Paradigm/Paradigm_CV_2024/Driving Pictures/Inputimages.png"

image = cv2.imread(turn_4)

# Process Image
processed_image = process_frame(image)

# Determine ROI
image_height = processed_image.shape[0]
image_width = processed_image.shape[1]
vertices = np.array([[(0, image_height), (image_width * 0.4, image_height * 0.6),
                      (image_width * 0.6, image_height * 0.6), (image_width, image_height)]], dtype=np.int32)
processed_image_roi = region_of_interest(processed_image, vertices)

# Performs a Hough Transform on the Image
hough_lines = hough_lines(processed_image_roi)
detected_lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
draw_lines2(detected_lines_image, hough_lines, color=[0, 255, 0], thickness=2)

# Averages lane lines
left_lane, right_lane = average_lines(hough_lines)

# Draw the averaged lane lines on the original image
lane_lines = []
if left_lane is not None:
    lane_lines.append([(int((image_height - left_lane[1]) / left_lane[0]), image_height),
                       (int((image_height * 0.6 - left_lane[1]) / left_lane[0]), int(image_height * 0.6))])

if right_lane is not None:
    lane_lines.append([(int((image_height - right_lane[1]) / right_lane[0]), image_height),
                       (int((image_height * 0.6 - right_lane[1]) / right_lane[0]), int(image_height * 0.6))])

lane_lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
draw_lines(lane_lines_image, lane_lines, color=[0, 0, 255], thickness=10)

# Weighted Image
processed_image_weighted = weighted_img(lane_lines_image, image)

# Display Original Image
image_resized = cv2.resize(image, (600, 400))
cv2.imshow("Original Image (Press 'q' to close)", image_resized)

# Display Detected Lines
detected_lines_image_resized = cv2.resize(detected_lines_image, (600, 400))
cv2.imshow("Detected Lines (Press 'q' to close)", detected_lines_image_resized)

#Display Processed Image
processed_image_weighted_resized = cv2.resize(processed_image_weighted, (600, 400))

# Add text to the top-left corner
text = (recommend_direction(lane_lines))
cv2.putText(processed_image_weighted_resized, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imshow("Processed Image (Press 'q' to close)", processed_image_weighted_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()