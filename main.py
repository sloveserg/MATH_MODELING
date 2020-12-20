from imageai.Detection import ObjectDetection
import os

exec_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(
	exec_path, "resnet50_coco_best_v2.0.1.h5")
)
detector.loadModel()
custom_objects = detector.CustomObjects(car=True, motorcycle=True,person=True, bicycle=True, airplane=True)
list = detector.detectCustomObjectsFromImage(
    custom_objects=custom_objects,
	input_image=os.path.join(exec_path, "objects.jpg"),
	output_image_path=os.path.join(exec_path, "new_objects.jpg"),
	minimum_percentage_probability=20,
	display_percentage_probability=True,
	display_object_name=True,

)
detections, objects_path = detector.detectObjectsFromImage(input_image=os.path.join(exec_path , "objects.jpg"), output_image_path=os.path.join(exec_path , "objects_separetely.jpg"), minimum_percentage_probability=30,  extract_detected_objects=True)

for eachObject, eachObjectPath in zip(detections, objects_path):
    print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("Object's image saved in " + eachObjectPath)
    print("--------------------------------")