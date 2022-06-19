# limit the number of cpus used by high performance libraries
from datetime import datetime
import os
import numpy as np


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, IMG_FORMATS, VID_FORMATS
from yolov5.utils.general import LOGGER, check_img_size, check_file, non_max_suppression, scale_coords, xyxy2xywh, increment_path
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

parser = argparse.ArgumentParser()
parser.add_argument('--yolo_model', nargs='+', type=str, default='assets/model-v2.pt', help='model.pt path(s)')
parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
parser.add_argument('--config_deepsort', type=str, default="deep_sort/configs/deep_sort.yaml")
parser.add_argument('--track', action='store_true', help='turn on and off tracking model')
parser.add_argument('--source', type=str, default='data/images/IP-0000002.png', help='source')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--show', action='store_true', help='display tracking video results')
parser.add_argument('--save', action='store_true', help='save video tracking results')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
parser.add_argument('--agnostic-nms', action='store_false', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

opt = parser.parse_args()

def log_msg(msg):
    print("{}: {}".format(datetime.now(),msg))

def predict(opt):
    yolo_model, show, save, imgsz, project, exist_ok= opt.yolo_model, opt.show, opt.save, opt.imgsz, opt.project, opt.exist_ok

    if opt.track:
        from deep_sort.utils.parser import get_config
        from deep_sort.deep_sort import DeepSort
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        deepsort = DeepSort(opt.deep_sort_model,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=True)

    is_file = Path(opt.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    
    webcam = opt.source == '0'
    
    if is_file:
        source = check_file(source)
    # Initialize
    device = select_device(opt.device)
    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Set Dataloader
    vid_path, vid_writer = None, None

    # Dataloader
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for (path, img, im0s, vid_cap, s) in dataset:
        t1 = time_sync()
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference

        pred = model(img, augment=opt.augment)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}'
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += ' Input size: %gx%g\n' % img.shape[2:]  # print string
            
            annotator = Annotator(im0, pil=not ascii)
            w, h = im0.shape[1],im0.shape[0]
            counts = {}
            results = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                
                s += "Object detections: "
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}" if c == det[:, -1].unique()[-1] else f"{n} {names[int(c)]}, "# add to string
                    if names[int(c)] not in counts:
                        counts[names[int(c)]] = int(n)

                if opt.track and dataset.mode != 'image':
                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]
                    t4 = time_sync()
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4
                
                    if len(outputs) > 0:
                        for j, output in enumerate(outputs):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            c = int(cls)  # integer class
                            label = f'{id} {names[c]}'
                            annotator.box_label(bboxes, label, color=colors(c, True))
                    # LOGGER.info(f'{s}\nInference time: YOLO: t3 - t2:.3f}s - Deep SORT: {t5 - t4:.3f}s')
                else:
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label =f'{names[c]}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        results.append({
                            'labelId': c,
                            'labelName': label,
                            'color': colors(c, True),
                            'probability': int(conf),
                            'boundingbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]),int(xyxy[3])]
                        })
                    LOGGER.info(f'{s}\nInference time: YOLO: {t3 - t2:.3f}s')
            else:
                if opt.track and dataset.mode != 'image':
                    deepsort.increment_ages()
                LOGGER.info('No detections')

            response = {
                'created': datetime.utcnow().isoformat(),
                'predictions': results if results else "No object detected",
                'count': counts if counts else "No object detected"
            }
    
            # Stream results
            im0 = annotator.result()
            if show:
                if dataset.mode == 'image' and opt.track is False:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(0)
                else:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
                        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape, %.1fms deep sort tracking {(1, 3, *imgsz)}' % t)
                        sys.exit()
    
            # Save results (image with detections)
            if save:
                save_dir = increment_path(Path(project) / "image" if dataset.mode == 'image' else "video", exist_ok=exist_ok)  # increment run
                save_dir.mkdir(parents=True, exist_ok=True)  # make dir
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                if dataset.mode == 'image' and opt.track is False:
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
                        
                    vid_writer.write(im0)
    if dataset.mode == 'video':
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'FPS: {1000/sum(t):.1f}')
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape, %.1fms deep sort update {(1, 3, *imgsz)}' % t)
    if save:
        print('Results saved to %s' % save_path)
    return response

def predict_image(img0):
    device = select_device(opt.device)
    # Load model
    model = DetectMultiBackend(opt.yolo_model, device=device, dnn=opt.dnn)
    
    img = letterbox(img0, 640, stride=32, auto=True)[0]
    
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    names = model.names
      
    pred = model(img, augment=opt.augment)
    im0 = img0.copy()
    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)

    # Process detections
    for det in pred:  # detections per image        
        annotator = Annotator(im0, pil=not ascii)
        
        counts = {}
        results = []
        if det is not None and len(det):
            # Rescale boxes from img_size to img size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()
            
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                if names[int(c)] not in counts:
                    counts[names[int(c)]] = int(n)
                    
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label =f'{names[c]}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                results.append({
                    'labelId': c,
                    'labelName': label,
                    'color': colors(c, True),
                    'probability': conf.numpy(),
                    'boundingbox': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]),int(xyxy[3])]
                })

        response = {
            'created': datetime.utcnow().isoformat(),
            'predictions': results if results else "No object detected",
            'count': counts if counts else "No object detected"
        }
    return response

if __name__ == "__main__":
    from PIL import Image
    image = np.array(Image.open("data/images/IP-0000001.png"))
    response = predict_image(image)