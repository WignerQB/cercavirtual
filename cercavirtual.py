"""
-----> COMANDOS PARA RODAR O PROGRAMA
---> Utilizando o vídeo vid8.mp4
--> python virtual_fence.py --weights ./checkpoints/yolov4-416 --score 0.3 --video ./data/vid8.mp4 --output ./results/demo.avi --model yolov4
-> (XlineINICIO, YlineINICIO) = (200, 240)
-> (XlineFIM, YlineFIM) = (1140, 100)

--->Utilizando o vídeo vid#.mp4
--> python virtual_fence.py --
hts ./checkpoints/yolov4-416 --score 0.3 --video ./data/vid#.mp4 --output ./results/demo.avi --model yolov4
-> (XlineINICIO, YlineINICIO) = (#, #)
-> (XlineFIM, YlineFIM) = (#, #)
"""

import csv
import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import glob
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
#flags.DEFINE_string('weights', './checkpoints/yolov4-416','path to weights file')
#flags.DEFINE_string('weights', './checkpoints/yolov4-416','/data/yolov4.weights')
flags.DEFINE_string('weights', './checkpoints/yolov4-416','/data/yolov4-tiny.weights')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


#--------------------------------------------------------------------------------------------------
print("TF version= ",tf.__version__)
print("np version= ",np.__version__)
print("Teste rodou!")
#--------------------------------------------------------------------------------------------------





#Parâmetros para construção das linhas----------------------------------------------------------------------------------------------
XlineINICIO = 200
YlineINICIO = 240
XlineFIM = 1140
YlineFIM = 100
Xperson = 200
Yperson = 240
CORlinhaP = (255, 145, 0) #Cor da linha principal sem detecção
CORlinha2 = (255, 255, 0) #Cor da linha secundárias sem detecção
CORlinha3 = (255, 255, 0) #Cor da linha secundárias sem detecção
CORlinha4 = (255, 255, 0) #Cor da linha secundárias sem detecção
CORlinha5 = (255, 255, 0) #Cor da linha secundárias sem detecção
Colorperson = CORlinhaP
OFFsetLinha24 = 56 #Offset das linhas 2 e 4
OFFsetLinha35 = OFFsetLinha24*2 #Offset das linhas 3 e 5
m = (YlineFIM-YlineINICIO)/(XlineFIM-XlineINICIO)
m2 = -1/m
List_person_inv = []
ListReadFile = []
LowerValue = 10000000000
Yprojected = Yperson
Xprojected = Xperson

#Calcular a distância de um ponto a uma reta
Xaux = int((XlineINICIO + XlineFIM)/2)
Yaux = int((Xaux - XlineINICIO)*m + YlineINICIO + OFFsetLinha24)
dist = int(abs(-m*Xaux + 1*Yaux + (m*XlineINICIO - YlineINICIO))/(pow((pow(-m, 2)+pow(1, 2)) , 0.5)))

#Função para criar dados sobre as pessoas rastreadas
class Invader():
    def __init__(self, classname,id, color_person, Xinvader, Yinvader, InvasionStatus, InvasionTime):
        self.classname = classname
        self.id = id
        self.color_person = color_person
        self.Xinvader = Xinvader
        self.Yinvader = Yinvader
        self.InvasionStatus = InvasionStatus
        self.InvasionTime = InvasionTime


#Função para desenhar as 5 linhas da cerca virtual
def LineBuild(image_frame, Xperson, Yperson, Colorperson):
    cv2.line(image_frame, (XlineINICIO, YlineINICIO), (XlineFIM, YlineFIM), CORlinhaP, 3)  # Linha principal
    cv2.line(image_frame, (XlineINICIO, YlineINICIO + OFFsetLinha24), (XlineFIM, YlineFIM + OFFsetLinha24), CORlinha2, 2)  # Linha 2
    #cv2.line(image_frame, (XlineINICIO, YlineINICIO + OFFsetLinha35), (XlineFIM, YlineFIM    + OFFsetLinha35), CORlinha3, 2)  # Linha 3
    cv2.line(image_frame, (XlineINICIO, YlineINICIO - OFFsetLinha24), (XlineFIM, YlineFIM - OFFsetLinha24), CORlinha4, 2)  # Linha 4
    #cv2.line(image_frame, (XlineINICIO, YlineINICIO + OFFsetLinha35), (XlineFIM, YlineFIM + OFFsetLinha35), CORlinha5, 2)  # Linha 5
    #cv2.line(image_frame, (XlineINICIO, YlineINICIO), (Xperson, Yperson), Colorperson, 2)  # Linha de detecção
    
#------------------------------------------------------------------------------------------------------------------------------------
#Função para detectar se houve alguma invasão
def ZonaMonitoramento(IMG, Class_Name, Track_Id , Colpers, Xprsn, Yprsn):
    #print(Class_Name + "-" + str(Track_Id) + " está na zona de monitoramento")    


    num = (Xprsn/m) + Yprsn + XlineINICIO*m - YlineINICIO
    den = (pow(m, 2) + 1)/m
    Xprojected = num/den
    Yprojected = int(-(Xprojected/m) + (Xprsn/m) + Yprsn+ OFFsetLinha24)
    Xprojected = int(Xprojected)
    m2 = (Yprsn - Yprojected)/(Xprsn - Xprojected)
    #print(m,m2,m*m2)

    #(Yprojected, Xprojected) -> É o ponto referente a pessoa projetado em cima da cerca
    cv2.line(IMG, (Xprsn, Yprsn), (Xprojected, Yprojected), (100, 25, 100), 2)
    m2 = (Yprsn - Yprojected)/(Xprsn - Xprojected)
    dist = pow((pow((Xprojected - Xprsn), 2)+pow((Yprojected - Yprsn), 2)),0.5)  
    

    #print(dist, "<", OFFsetLinha24)
    if dist < OFFsetLinha24:
        cv2.putText(IMG,"Houve uma invasao",(140, 500),0, 2.0, (255,0,0),10)
        print("Houve uma invasão")
        #cv2.line(IMG, (List_person_inv[0].Xinvader, List_person_inv[0].Yinvader), (X_end, Y_end), (100, 25, 100), 2)
        IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
        
        Pessoa = 'Pessoa {}'.format(Track_Id)
        #HorarioInv = '{}'.format(DateTimeNow.strftime("%X"))
        HorarioInv = "12:54"
        #DataInv = '{}'.format(DateTimeNow.strftime("%x"))
        DataInv = "07/07/2022"
        IdCam = 3
        
        Invaders = []
        
        #print("Entrei aqui")
        
        rows = []
        with open('./capturas/Historico.csv','r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)
            for row in csvreader:
                rows.append(row)
        
        for row in rows:
            print(row[1])
        
        
        with open('./capturas/Historico.csv','a', newline='') as csvfile:
            fieldnames = ['Invasor','Horario da invasao','Data da invasao','Id da camera']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #writer.writeheader()
            #writer.writerows(Invaders)
            writer.writerow({'Invasor':Pessoa,'Horario da invasao':HorarioInv,'Data da invasao':DataInv,'Id da camera':IdCam})
        
        for index in enumerate(glob.glob('./capturas/*.*')):
            pass
        print("o index eh: ", index[0])
        index = index[0]
        #Salva imagem da pessoa que invadiu
        stt = cv2.imwrite('./capturas/Person{}:{}.png'.format(Track_Id,index), IMG)            
                    
        
"""
        f = open("C:/Users/josew/Pictures/Saved Pictures/Informação sobre a invasão.txt", "r")
        ListReadFile.clear()
        for ReadFile in f:
            ListReadFile.append(ReadFile)
        f.close()

        #print("\n\n",bool(ListReadFile),"\n\n")

        
        #print(ListReadFile)
        #print("$$$$$$$$$$$$$$$$$$$$$$")
                

        #Obtem informação da data atual
        DateTimeNow = datetime.datetime.now()

        if bool(List_person_inv):#Verifica se List_person_inv não está vazia
            #Percorre a List_person_inv para analisar se o indivíduo X já está na lista de pessoas que invadiram
            for listofpersoninv in List_person_inv:
                #print("\n To no for 1")
                if bool(ListReadFile):
                    #print(ListReadFile)
                    for pointer in ListReadFile:
                        #print("To no for 2", pointer)

                        i = pointer.rindex(" invadiu!")
                        if int(pointer[6:i]) == listofpersoninv.id:
                            print("Pessoa {} já invadiu".format(int(pointer[6:i])))
                        else:
                            f = open("C:/Users/josew/Pictures/Saved Pictures/Informação sobre a invasão.txt", "a")
                            f.writelines(["Pessoa {} invadiu!".format(Track_Id), "\tHorário da invasão: {}".format(DateTimeNow.strftime("%X")), "\tData da invasão: {}".format(DateTimeNow.strftime("%x")), "\tCâmera: 1\n"])
                            f.close()
                            for index in enumerate(glob.glob('C:/Users/josew/Pictures/Saved Pictures/*.*')):
                                pass
                            #print(index)
                            index = index[0]
                            #Salva imagem da pessoa que invadiu
                            stt = cv2.imwrite('C:/Users/josew/Pictures/Saved Pictures/Test{}.png'.format(index), IMG)
                            #print("Break do for 2")
                            break
                else:
                    f = open("C:/Users/josew/Pictures/Saved Pictures/Informação sobre a invasão.txt", "a")
                    f.writelines(["Pessoa {} invadiu!".format(Track_Id), "\tHorário da invasão: {}".format(DateTimeNow.strftime("%X")), "\tData da invasão: {}".format(DateTimeNow.strftime("%x")), "\tCâmera: 1\n"])
                    f.close()
                    for index in enumerate(glob.glob('C:/Users/josew/Pictures/Saved Pictures/*.*')):
                        pass
                    #print(index)
                    index = index[0]
                    #Salva imagem da pessoa que invadiu
                    stt = cv2.imwrite('C:/Users/josew/Pictures/Saved Pictures/Test{}.png'.format(index), IMG)
                    #print("Break do for 1")
                    break




                if listofpersoninv.id == Track_Id:#Se o indivíduo estiver na lista, é feito a atualização dos outros parâmetros
                                                  #deste indivíduo
                    List_person_inv.remove(listofpersoninv)
                #Caso contrário é somente adicionado
                List_person_inv.append(Invader(Class_Name, Track_Id, Colpers, Xprsn, Yprsn, "TRUE", DateTimeNow.strftime("%X")))
                

                
            print("-----------------------")
        else:
            for listofpersoninv in List_person_inv:
                if bool(ListReadFile):
                    for pointer in ListReadFile:
                        i = pointer.rindex(" invadiu!")
                        if int(pointer[6:i]) == listofpersoninv.id:
                            print("Pessoa {} já invadiu".format(int(pointer[6:i])))
                        else:
                            f = open("C:/Users/josew/Pictures/Saved Pictures/Informação sobre a invasão.txt", "a")
                            f.writelines(["Pessoa {} invadiu!".format(Track_Id), "\tHorário da invasão: {}".format(DateTimeNow.strftime("%X")), "\tData da invasão: {}".format(DateTimeNow.strftime("%x")), "\tCâmera: 1\n"])
                            f.close()
                else:
                    f = open("C:/Users/josew/Pictures/Saved Pictures/Informação sobre a invasão.txt", "a")
                    f.writelines(["Pessoa {} invadiu!".format(Track_Id), "\tHorário da invasão: {}".format(DateTimeNow.strftime("%X")), "\tData da invasão: {}".format(DateTimeNow.strftime("%x")), "\tCâmera: 1\n"])
                    f.close()

            List_person_inv.append(Invader(Class_Name, Track_Id, Colpers, Xprsn, Yprsn, "TRUE", DateTimeNow.strftime("%X")))
            


    else:
        cv2.line(IMG, (List_person_inv[0].Xinvader, List_person_inv[0].Yinvader), (X_end, Y_end), Colpers, 2)
"""





def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        #print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        Xperson = 1
        Yperson = 1
        Colorperson = (255, 255, 255)
        

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            Xperson = int(bbox[0]) + int((bbox[2] - bbox[0])/2)
            Yperson = int(bbox[1]) + int((bbox[3] - bbox[1])/2)
            Colorperson = color
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

	
        LineBuild(frame, Xperson, Yperson, Colorperson)
        try:
            YCheck = int(m*(Xperson - XlineINICIO) + YlineINICIO)
            #print("A posiçao da pessoa ", track.track_id," eh: ",(Xperson,YCheck))
            if Yperson > (YCheck - OFFsetLinha24) and Yperson < (YCheck + OFFsetLinha24): #Entre as linhas 4 e 2
                print("Zona da cerca")
                ZonaMonitoramento(frame, class_name, track.track_id, Colorperson, Xperson, Yperson)
        except:
            pass
	
	


        # calculate frames per second of running detections
        #fps = 1.0 / (time.time() - start_time)
        #print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            pass
            #cv2.imshow("Output Video", result)
        
    # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
