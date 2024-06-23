import argparse
import time
import os
from pathlib import Path
import sys
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import pyzed.sl as sl
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def detect(save_img=False):
    # Erstelle ein ZED-Kameraobjekt
    zed = sl.Camera()

    # Erstelle die Initialisierungsparameter
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Wählen Sie die Auflösung
    init_params.camera_fps = 30  # Wählen Sie die FPS

    # Öffne die Kamera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Fehler beim Öffnen der Kamera: {err}")
        exit(1)

    print("[INFO] Starting streaming...")
    print("[INFO] Camera ready.")
    detections_file_path = Path("detections.txt")
    weights, view_img, save_txt, imgsz = (
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
    )

    # Directories
    save_dir = Path(f"{Path(opt.project)} + / {opt.name}")  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(
        weights=os.path.join(os.getcwd(), "yolov5s.pt"), map_location=device
    )  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    webcam = False
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )  # run once
    t0 = time.time()
    while True:
        save_img = True
        path = os.getcwd()

        # Hole das Bild und die Tiefenkarte
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Hole das Bild und die Tiefenkarte
            image = sl.Mat()
            depth = sl.Mat()
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

            color_image = image.get_data()
            depth_data = depth.get_data()
            im0s = color_image
            img = letterbox(im0s)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred,
                opt.conf_thres,
                opt.iou_thres,
                classes=opt.classes,
                agnostic=opt.agnostic_nms,
            )
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = (
                        path[i],
                        "%g: " % i,
                        im0s[i].copy(),
                        dataset.count,
                    )
                else:
                    p, s, im0 = path, "", im0s  # , getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += "%gx%g " % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape
                    ).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                                .view(-1)
                                .tolist()
                            )  # normalized xywh
                            line = (
                                (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                            )  # label format
                            with open(txt_path + ".txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

                        if save_img or view_img:  # Add bbox to image
                            label = f"{names[int(cls)]} {conf:.2f}"
                            plot_one_box(
                                xyxy,
                                im0,
                                label=label,
                                color=colors[int(cls)],
                                line_thickness=3,
                            )
                            #############################################
                            d1, d2 = int((int(xyxy[0]) + int(xyxy[2])) / 2), int(
                                (int(xyxy[1]) + int(xyxy[3])) / 2
                            )
                            zDepth = depth_data[d2, d1]  # Depth in millimeters
                            zDepth_cm = zDepth / 10  # Convert to centimeters
                            tl = 3  # line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                            tf = max(tl - 1, 1)  # font thickness
                            with open(detections_file_path, "a") as f:
                                f.write(
                                    f"{names[int(cls)]} {conf:.2f} {round(zDepth_cm, 2)} cm\n"
                                )  # writes the label, the confidence, and the distance in cm to file
                            cv2.putText(
                                im0,
                                f"{round(zDepth_cm, 2)} cm",
                                (d1, d2),
                                0,
                                tl / 3,
                                [225, 255, 255],
                                thickness=tf,
                                lineType=cv2.LINE_AA,
                            )
                            # print(zDepth, d1, d2)
                            #############################################
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord("q"):
                break

    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        print(f"Results saved to {save_dir}{s}")

    # Close the camera
    zed.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="yolov7.pt", help="model.pt path(s)"
    )
    # parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    opt = parser.parse_args()
    print(opt)
    # check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
