import numpy as np
import cv2

def display_matches(image, boxes, names, box_color=(0, 255, 0),unknown_color=(0, 0, 255), font_scale=0.7,
thickness=2, window_name="Face Recognition"):
    """
    Parameters: 
    imgage: the picture loaded in as a numpy array (H, W, 3)
    boxes: has a shape of (N,4), N is the number of faces detected and each row is 4 numbers (x1, y1, x2, y2)
    names=name of the people detected

    Other paramaeters are defined but include:
    box_color=(0, 255, 0) (if person known)
    unknown_color=(0, 0, 255) (if we dont know the person)

    Returns: the picture (with bound boxes) back to Python as a NumPy array.
    It also opens a pop up window so we can see the photo with all the bounding boxes
    
    """
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # turns image from bgr to rgb
    if boxes is None or len(boxes) == 0:
        cv2.imshow(window_name, img_bgr)
        return image
    
    boxes = np.asarray(boxes)
    if len(boxes) != len(names):
        raise ValueError("The number of boxes doesn't match the number of names")
    
    for box, name in zip(boxes, names):
        x1, y1, x2, y2 = box.astype(int)
        color = unknown_color if name == "Unknown" else box_color

        # draw the bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

        # measure text size 
        (text_w, text_h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10

        cv2.rectangle(
            img_bgr,
            (x1, label_y - text_h - 4),
            (x1 + text_w + 4, label_y + 4),
            color, -1  
        )

        cv2.putText(
            img_bgr, name, (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), thickness
        )

    cv2.imshow(window_name, img_bgr)
    cv2.waitKey(0)  # keeps window open until a key is pressed
    cv2.destroyAllWindows()
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
