import colorsys
import os
from timeit import default_timer as timer
import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image
import darknet as darknet

treeAreaList = []
planetAreaList = []
beachAreaList = []


class ClassesList(object):
    def __init__(self):
        # List data example:[0,[1,2,3,4]] = [area,[xmin,ymin,xmax,ymax]]
        self.treeAreaList = []
        self.planetAreaList = []
        self.beachAreaList = []

    def CalculateOverlap(self, left_top, right_buttom):
        xmax = right_buttom[0]
        ymax = right_buttom[1]
        xmin = left_top[0]
        ymin = left_top[1]
        # if is overlap
        if xmax > xmin and ymax > ymin:
            intersectionArea = (xmax-xmin) * (ymax-ymin)
            return intersectionArea
        else:
            return 0

    def CalculateUnion(self, _list):
        totalArea = float(0.0)
        for data in _list:
            totalArea += data[0]
        print(totalArea)
        i = 0
        j = 0
        totalIntersection = float(0.0)
        while i < len(_list):
            j = i+1
            while j < len(_list):
                # A,B is two boxes that been determine
                xminA, yminA, xmaxA, ymaxA = _list[i][1][0], _list[i][1][1], _list[i][1][2], _list[i][1][3]
                xminB, yminB, xmaxB, ymaxB = _list[j][1][0], _list[j][1][1], _list[j][1][2], _list[j][1][3]
                left_top = (max(xminA, xminB), max(yminA, yminB))
                right_buttom = (min(xmaxA, xmaxB), min(ymaxA, ymaxB))
                print(left_top)
                print(right_buttom)
                totalIntersection += self.CalculateOverlap(
                    left_top, right_buttom)
                j += 1
            i += 1
        totalUnion = totalArea-totalIntersection
        return totalUnion


def _convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def CalculateBoxesArea(xmin, ymin, xmax, ymax):
    """
    calculate the area of a object
    """
    boxArea = (xmax-xmin)*(ymax-ymin)
    return boxArea


def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


class YOLO(object):
    _defaults = {
        "configPath": "./cfg/yolo-obj.cfg",
        "weightPath": "./backup/yolo-obj_8000.weights",
        "metaPath": "./data/obj.data",
        "classes_path": "./data/obj.names",
        "thresh": 0.25,
        "iou_thresh": 0.5,
        # "model_image_size": (416, 416),
        # "model_image_size": (608, 608),
        "model_image_size": (800, 800),
        "gpu_num": 1,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.colors = self._get_colors()
        self.netMain = darknet.load_net_custom(self.configPath.encode("ascii"), self.weightPath.encode("ascii"), 0,
                                               1)  # batch size = 1
        self.metaMain = darknet.load_meta(self.metaPath.encode("ascii"))
        self.altNames = self._get_alt_names()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, encoding="utf-8") as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_colors(self):
        class_names = self._get_class()
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        # Shuffle colors to decorrelate adjacent classes.
        np.random.shuffle(colors)
        np.random.seed(None)  # Reset seed to default.
        return colors

    def _get_alt_names(self):
        try:
            with open(self.metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
        return altNames

    def cvDrawBoxes(self, detections, image):
        # 字型相關設定，包括字型檔案路徑、字型大小
        font = ImageFont.truetype(font='font/simfang.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 檢測框的邊框厚度，該公式使得厚度可以根據圖片的大小來自動調整
        thickness = (image.size[0] + image.size[1]) // 300  #
        # 遍歷每個檢測到的目標detection:(classname,probaility,(x,y,w,h))
        for c, detection in enumerate(detections):
            # 獲取當前目標的類別和置信度分數
            classname = detection[0]
            if classname == 'b"tree"':
                print("111111111111111111111111111111111")
            # score = round(detection[1] * 100, 2)
            score = round(float(detection[1]), 2)
            label = '{} {:.2f}'.format(classname, score)
            # 計算檢測框左上角(xmin, ymin)和右下角的座標(xmax, ymax)
            x, y, w, h = detection[2][0], \
                detection[2][1], \
                detection[2][2], \
                detection[2][3]
            xmin, ymin, xmax, ymax = _convertBack(
                float(x), float(y), float(w), float(h))
            # 獲取繪製例項
            draw = ImageDraw.Draw(image)
            # 獲取將顯示的文字的大小
            label_size = draw.textsize(label, font)
            # 將座標對應到top, left, bottom, right，注意不要對應錯了
            top, left, bottom, right = (ymin, xmin, ymax, xmax)
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            if c > len(self.class_names) - 1:
                c = 1
            # 繪製邊框厚度
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            # 繪製檢測框的文字邊界
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            # 繪製文字
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
            # [TODO] add area of detection to list
            area = CalculateBoxesArea(xmin, ymin, xmax, ymax)
            if str(classname) == "b\'tree\'":
                treeAreaList.append([area, (xmin, ymin, xmax, ymax)])
            elif str(classname) == "b\'planet\'":
                planetAreaList.append(
                    [area, (xmin, ymin, xmax, ymax)])
            elif str(classname) == "b\'beach\'":
                beachAreaList.append(
                    [area, (xmin, ymin, xmax, ymax)])
        return image

    def detect_video(self, video_path, output_path, show=True):
        nw = self.model_image_size[0]
        nh = self.model_image_size[1]
        assert nw % 32 == 0, 'Multiples of 32 required'
        assert nh % 32 == 0, 'Multiples of 32 required'
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (nw, nh)
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(
                video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(
                output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()

        # Create an image we reuse for each detect
        darknet_image = darknet.make_image(nw, nh, 3)
        while True:
            return_value, frame = vid.read()
            if return_value:
                # 轉成RGB格式，因為opencv預設使用BGR格式讀取圖片，而PIL是用RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image_resized = image.resize(video_size, Image.LINEAR)
                darknet.copy_image_from_bytes(
                    darknet_image, np.asarray(image_resized).tobytes())
                detections = darknet.detect_image(self.netMain, self.metaMain, darknet_image,
                                                  thresh=self.thresh)
                image_resized = self.cvDrawBoxes(detections, image_resized)
                result = np.asarray(image_resized)

                # 轉成BGR格式以便opencv處理
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                curr_time = timer()
                exec_time = curr_time - prev_time
                prev_time = curr_time
                accum_time = accum_time + exec_time
                curr_fps = curr_fps + 1
                if accum_time > 1:
                    accum_time = accum_time - 1
                    fps = "FPS: " + str(curr_fps)
                    curr_fps = 0
                cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)
                if show:
                    cv2.imshow("Object Detect", result)
                if isOutput:
                    print("start write...==========================================")
                    out.write(result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        out.release()
        vid.release()
        cv2.destroyAllWindows()
        # [TODO] calculate data and display
        instance = ClassesList()
        CalculateUnion = instance.CalculateUnion
        print(CalculateUnion(treeAreaList))
        print(CalculateUnion(planetAreaList))
        print(CalculateUnion(beachAreaList))
        print("video write complete")

    def detect_image(self, image_path, save_path):
        nw = self.model_image_size[0]
        nh = self.model_image_size[1]
        assert nw % 32 == 0, 'Multiples of 32 required'
        assert nh % 32 == 0, 'Multiples of 32 required'
        try:
            image = Image.open(image_path)
        except:
            print('Open Error! Try again!')
        else:
            image_resized = image.resize((nw, nh), Image.LINEAR)
            darknet_image = darknet.make_image(nw, nh, 3)
            darknet.copy_image_from_bytes(
                darknet_image, np.asarray(image_resized).tobytes())
            # 識別圖片得到目標的類別、置信度、中心點座標和檢測框的高寬
            detections = darknet.detect_image(self.netMain, self.metaMain, darknet_image,
                                              thresh=self.thresh)
            # detections = darknet.detect_image(self.class_names, self.netMain, self.metaMain, darknet_image,
            #                                   thresh=0.25, debug=True)
            # 在圖片上將detections資訊繪製出來
            image_resized = self.cvDrawBoxes(detections, image_resized)
            # 顯示繪製後的圖片
            image_resized.show()
            image_resized.save(save_path)


# if __name__ == "__main__":
#     _yolo = YOLO()
#     _yolo.detect_image("static/img/11111.jpg",
#                        "static/img/boundedImg/11111.JPG")
    # _yolo.detect_video("./data/testVideo.mp4",
    #                    "./Demo/test_output.avi", show=False)
