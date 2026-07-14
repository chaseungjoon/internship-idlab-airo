"""simple image viewer for homework 4 that prints the coordinates of the mouse click. 
should be used to collect the coordinates of corresponding points in the left and right images."""
import cv2
import pathlib 

DATA_DIR = pathlib.Path(__file__).parent / 'data_0000'
left_image_path = DATA_DIR / "image_left.png"
right_image_path = DATA_DIR / "image_right.png"

left_image = cv2.imread(str(left_image_path))
right_image = cv2.imread(str(right_image_path))

left_point = [None]
right_point = [None]

cyan = (255, 255, 0) # color for left image
magenta = (255, 0, 255) # color for right image


def on_click_left(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"left click: (u,v) =  ({x}, {y})")
        left_point[0] = (x, y)

def on_click_right(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"right click: (u,v) =  ({x}, {y})")
        right_point[0] = (x, y)


cv2.namedWindow("Left Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Left Image", on_click_left)
cv2.setMouseCallback("Right Image", on_click_right)

# resize the windows and place them side by side
cv2.resizeWindow("Left Image", 800, 600)
cv2.resizeWindow("Right Image", 800, 600)
cv2.moveWindow("Left Image", 0, 0)
cv2.moveWindow("Right Image", 800, 0)

def draw_point_coordinate(image, point, color):
    """Draws a circle on the image at the given point with the coordinates next to it."""
    text = f"({point[0]}, {point[1]})"
    text_location = (point[0] + 12, point[1] - 12)
    cv2.circle(image, point, 1, color, 1)
    cv2.circle(image, point, 15, color, 4)
    cv2.putText(image, text, text_location, cv2.FONT_HERSHEY_SIMPLEX,  1.5, color, 3, cv2.LINE_AA)

while True:
    left_annotated = left_image.copy()
    right_annotated = right_image.copy()

    if left_point[0] is not None:
        draw_point_coordinate(left_annotated, left_point[0], cyan)

    if right_point[0] is not None:
        draw_point_coordinate(right_annotated, right_point[0], magenta)

    cv2.imshow("Left Image", left_annotated)
    cv2.imshow("Right Image", right_annotated)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

print("Left point: ", left_point[0])
print("Right point: ", right_point[0])