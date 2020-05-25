import io
import os
import threading as th
import time

import MySQLdb as mysql
import PySimpleGUI as sg
import cv2
import numpy as np
from PIL import Image
from pybeep.pybeep import PyBeep
from concurrent.futures import ProcessPoolExecutor

sg.theme("DarkGrey3")
CONFIDENCE = 0.4
THRESHOLD = 0.3
scale = 1

yolo_dir = '/home/moby/PycharmProjects/opencv_exp/helmatDetect/darknet'
weightsPath = os.path.join(yolo_dir, 'yolov3-hat_9000.weights')
configPath = os.path.join(yolo_dir, 'cfg/yolov3-hat.cfg')
labelsPath = os.path.join(yolo_dir, 'data/hat.names')
no_hatPath = yolo_dir + '/face/'

Camera = 0
global fps
fps = 0
time_interval = 10
pool = ProcessPoolExecutor()


def warning():
    PyBeep().beep()
    PyBeep().beepn(0)


def writeImg(nowTime, img):
    cv2.imwrite(yolo_dir + '/face/' + nowTime + '.jpg', img)
    # time.sleep(5)


def saveImg(img):
    conn = mysql.connect('localhost', 'root', 'Song1997', 'faces')
    cursor = conn.cursor()
    fp = open(img, 'rb')
    img = fp.read()
    sql = "INSERT INTO no_hat (face) VALUES (%s)"
    cursor.execute(sql, [img])
    conn.commit()
    cursor.close()
    conn.close()


def update2Sql_and_delRaw():
    for path, dir_list, file_list in os.walk(no_hatPath):
        for file_name in file_list:
            # print(os.path.join(path, file_name))
            saveImg(os.path.join(path, file_name))
            os.remove(os.path.join(path, file_name))


def showImg(idx):
    conn = mysql.connect('localhost', 'root', 'Song1997', 'faces')
    cursor = conn.cursor()
    sql = "SELECT face FROM no_hat WHERE id=(%s)"
    cursor.execute(sql, (idx,))
    fout = cursor.fetchone()[0]
    fout = io.BytesIO(fout)
    img = Image.open(fout)
    img.show()
    cursor.close()
    conn.close()


def getFaceNum():
    conn = mysql.connect('localhost', 'root', 'Song1997', 'faces')
    cursor = conn.cursor()
    sql = "SELECT id FROM no_hat;"
    fout = int(cursor.execute(sql))
    cursor.close()
    conn.close()
    # print(fout)
    return fout


def clearSQL():
    conn = mysql.connect('localhost', 'root', 'Song1997', 'faces')
    cursor = conn.cursor()
    sql = "TRUNCATE TABLE no_hat;"
    cursor.execute(sql)
    cursor.close()
    conn.close()


def deleteFace(idx):
    conn = mysql.connect('localhost', 'root', 'Song1997', 'faces')
    cursor = conn.cursor()
    sql = "DELETE FROM no_hat WHERE id=(%s)"
    cursor.execute(sql, (idx,))
    conn.commit()
    cursor.close()
    conn.close()


# 通过class name的文档获取class的名称
# 通过随机数初始化颜色列表
with open(labelsPath, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# 加载yolo权重文件与yolo文件
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# openCV使用gpu加速
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

sg.popup_quick_message('载入YOLO权重...', background_color='red', text_color='white')
video = os.path.join(yolo_dir, '1.mp4')
cap = cv2.VideoCapture(video)
# cap = cv2.VideoCapture(Camera)

# layer_names = net.getLayerNames()
face = 0
imgflip = 1
Det_Num = 1
win_Start = False

while cap.isOpened():

    pool.submit(update2Sql_and_delRaw)
    s = th.Thread(target=warning, name='warning')

    ret, frame = cap.read()
    frame = cv2.flip(frame, flipCode=imgflip)
    frame_id = 0
    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    new_frame = frame.copy()

    # 将图片构建成一个blob，设置图片尺寸，然后执行一次
    # YOLO前馈网络计算，最终获取边界框和相应概率
    blobImg = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (416, 416),
                                    None, True, False)
    net.setInput(blobImg)
    outInfo = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(outInfo)
    (H, W) = frame.shape[:2]

    # 初始化边界框, 置信度, 类别
    boxes = []
    confidences = []
    classIDs = []

    # 迭代每个输出层
    for out in layerOutputs:
        # 迭代每个检测
        for detection in out:
            # 提取类和置信值
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE:
                # 将边界框的坐标还原至与原图片相匹配,
                # 返回的是边界框的中心坐标以及边界框的宽度和高度
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # 使用非极大值抑制方法抑制弱、重叠边界框
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    def detectObj(i):
        # 获取bounding box的坐标
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.2f}".format(labels[classIDs[i]], confidences[i])
        center = (x + w // 4, y + h // 2)
        no_hat = new_frame[y: y + h, x: x + w]
        if classIDs[i] == 0:
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            s.run()
            now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
            global fps
            if fps % time_interval == 0:
                write = th.Thread(target=writeImg, args=(now, no_hat))
                write.start()
            fps += 1
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, "NO HELMET!", center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


    def detectObjs(idxs):
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.2f}".format(labels[classIDs[i]], confidences[i])
            center = (x + w // 4, y + h // 2)
            no_hat = new_frame[y: y + h, x: x + w]
            if classIDs[i] == 0:
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                warning()
                now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
                global fps
                if fps % time_interval == 0:
                    pool.submit(writeImg, now, no_hat)
                fps += 1
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(frame, "NO HELMET!", center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if len(idxs) > 0:
        if Det_Num == 1:
            detectObjs(idxs=idxs)
        elif Det_Num == 0:
            i = idxs.flatten()[0]
            detectObj(i)
        else:
            sg.popup('系统错误', grab_anywhere=True)
            assert "System Error"

    imgBytes = cv2.imencode('.png', frame)[1].tobytes()

    if not win_Start:
        win_Start = True
        layout = [
            [sg.Text("目标检测中:", (10, 1))],
            [sg.Image(data=imgBytes, key='__IMAGE__')],
            [
                sg.Text('置信值', size=(7, 1)),
                sg.Slider(range=(0, 10), orientation='h', resolution=1,
                          default_value=3, size=(15, 15), key='confidence'),
                sg.Text('阈值', size=(4, 1)),
                sg.Slider(range=(0, 10), orientation='h', resolution=1,
                          default_value=3, size=(15, 15), key='threshold')
            ],
            [
                sg.Button("获取数据库图片数量", key='getnum'),
                sg.Text('', key='faceNum', size=(4, 1)),
                sg.Button("清空数据库", key='clear'),
                sg.Text("id: ", size=(4, 1)),
                sg.InputText(size=(15,)),
                sg.Button("提交", key="show")
            ],
            [
                sg.Text("输入id删除图片", size=(14, 1)),
                sg.InputText(size=(15,)),
                sg.Button("提交", key="del_img"),
                sg.Text("单/多目标检测", size=(14, 1)),
                sg.Slider(range=(0, 1), orientation='h', resolution=1,
                          default_value=1, size=(5, 15), key='det_num')
            ],
            [
                sg.Exit("退出", key="Exit")
            ]
        ]
        win = sg.Window("安全帽检测", layout, default_button_element_size=(14, 1),
                        text_justification='right', auto_size_text=False, finalize=True)
        image_elem = win['__IMAGE__']
    else:
        image_elem.update(data=imgBytes)

    event, values = win.read(timeout=0)
    if event is None or event == 'Exit':
        break

    elif event == 'getnum':
        fout = getFaceNum()
        win.Element('faceNum').Update(fout, text_color='red')

    elif event == 'show':
        faceId = int(values[0])
        try:
            showImg(faceId)
        except:
            sg.popup('错误', '无效ID', grab_anywhere=True)

    elif event == 'del_img':
        faceId = int(values[1])
        try:
            deleteFace(faceId)
            sg.popup('删除成功', grab_anywhere=True)
        except:
            sg.popup('错误', '无效ID', grab_anywhere=True)

    elif event == 'clear':
        clearSQL()
        sg.popup('数据库已清空', grab_anywhere=True)

    CONFIDENCE = values['confidence'] // 10
    THRESHOLD = values['threshold'] // 10
    Det_Num = int(values['det_num'])

# showImg(1)
print("[INFO] cleaning up...")
win.close()
