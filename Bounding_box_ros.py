# from v:ision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenet_v2_ssd_lite import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
import traceback


bridge = CvBridge()

rospy.init_node("MODEL_BOOBA",anonymous=True)

model_path = "./mb2-ssd-lite-Epoch-120-Loss-1.5612082481384277(1).pth"

class_names = ["BACKGROUND", "GATE"]

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)

net.load(model_path)
net.to("cpu")

predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device="cpu")
# predictor.to("cpu")

fcent_x, fcent_y = 320, 240


def image_callback(msg):
    try:
        frame = bridge.imgmsg_to_cv2(msg,'bgr8')
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        boxes, labels, probs = predictor.predict(image, 10, 0.4)

        for i in range(boxes.size(0)):
            if probs[i] < 0.5:
                continue
            box = boxes[i, :]
            box = list(map(int, box))

            bcent_x = (box[0] + box[2]) // 2
            bcent_y = (box[1] + box[3]) // 2

            cv2.line(frame, (fcent_x, fcent_y), (bcent_x, bcent_y), (255,255,255), 2)

            x_err, y_err = bcent_x - fcent_x, bcent_y - fcent_y
            rospy.loginfo(f"X error : {x_err}\tY error : {y_err}")

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(frame, label, (box[0] + 20, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Gate Detection", frame)
        cv2.waitKey(1)

    except Exception as e:
       # rospy.logerr(f"Error processing image: {e}")
        print(traceback.print_stack())

rospy.Subscriber("/camera_sony/image_raw", Image, image_callback)

rospy.spin()

cv2.destroyAllWindows()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # Convert frame from BGR to RGB for prediction
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     boxes, labels, probs = predictor.predict(image, 10, 0.4)
#     for i in range(boxes.size(0)):
#         box = boxes[i, :]
#         box = list(map(int, box))
#         bcent_x, bcent_y = box[0] + box[2] // 2, box[1] + box[3] // 2
#         x_err, y_err = bcent_x - fcent_x, bcent_y - fcent_y
#         cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
#         label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
#         cv2.putText(frame, label, (box[0] + 20, box[1] + 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#     cv2.imshow("Gate BBox", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()
