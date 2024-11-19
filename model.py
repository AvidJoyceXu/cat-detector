import torch
from PIL import Image

# device setup
device = "cpu" 
if torch.cuda.is_available():
    device = torch.device("cuda")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS backend")

from ultralytics import YOLO

# Load a pre-trained YOLO model for object detection
object_detector = YOLO("yolo11n.pt")  # Replace with a more advanced YOLO model if needed

# Combine object detector with CLIP to detect cat 
def detect_cats(images_path):
    '''
    Traverse through all the object proposals and let CLIP determine whether it's a cat.
    API of YOLO ckpt: 
        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): The source of
                the image(s) to make predictions on. Can be a file path, URL, PIL image, numpy array, PyTorch
                tensor, or a list/tuple of these.
            stream (bool): If True, treat the input source as a continuous stream for predictions.
            **kwargs (Any): Additional keyword arguments to configure the prediction process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, each encapsulated in a
                Results object.
    '''
    # Load and preprocess the image
    images = [Image.open(path).convert("RGB") for path in images_path]
    results = object_detector(images)  # Detect objects in the image

    cat_boxes = []
    n = len(images_path)
    assert n == len(results)
    for i in range(n):
        result = results[i]
        # print(result)
        cats = []
        for box in result.boxes: 
            print(box)
            if box.cls[0] == 15: # 15: 'cat' label for Yolo v8n
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cats.append((x1, y1, x2, y2))
        cat_boxes.append(cats)

    return cat_boxes

import matplotlib.pyplot as plt
from PIL import ImageDraw

def draw_boxes(image_path, boxes):
    n = len(image_path)
    assert n == len(boxes)
    for i in range(n):
        image = Image.open(image_path[i]).convert("RGB")
        draw = ImageDraw.Draw(image)
        for (x1, y1, x2, y2) in boxes[i]:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        image.save(f"output_{i}.png")
    return image

# Example Usage
image_path = ["1.png", "2.png"] 
detected_boxes = detect_cats(image_path)
print(len(detected_boxes))

if detected_boxes:
    output_image = draw_boxes(image_path, detected_boxes)
    plt.imshow(output_image)
    plt.axis("off")
    plt.show()
else:
    print("Nothing")
