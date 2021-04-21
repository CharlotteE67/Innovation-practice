# Development of Asymmetric Competitive Games Based on Bluetooth Technology

#### 											--Final Inspection Report



​													Supervisor:Song Xuan

​					Team members:Jiang Yuchen,Chen Jiyuan,Huang Wenjie

## 1.Abstract

This is a report for the final inspection.Conclusions of the second inspection and work of final stage are concluded in passage.Futhermore,summary of this term and future work are written at the end of the passage.



## 2.Conclusions of 2nd Inspection

​	We are reminded that we are behind schedule and we need to focus on our aim which is to build up a hand motion detection system. We need to build up our deep learning model as soon as possible.



## 3.Literature Review

​	Based on the the literature before, we finally chose Bi-LSTM to apply on our system.

### 	A.Motion Prediction

​	In traditional deep learning model,the deep learning model reads in whole time sequence and makes a recognition. Sometimes the system can't make a correct recognition. How to make the system more accurate is an important part to optimize it. Thus,predicting movement is  given to make the model give more accurate result.[1]

![image-20201202001850994](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201202001850994.png)

​										Fig.1 Framework for action recognition and movement prediction[1]

​	As the figure above shows,another LSTM recurrent neural network is added which can assist initial network in giving recognition result with corresponding probability. At the same time,the initial network whose name is RegLSTM provide class probability as one of inputs for PredLSTM which represent prediction network.

​	Here is a figure which shows the result when applying PredLSTM. It's clear that when applying prediction module,recognition accuracy increases,especially for LSTM system. It can give a high-accuracy result when observation ratio is about 0.8 and also has a higher accuracy when whole observation is finished. Prediction model  performs better than traditional  LSTM recurrent neural network.

![image-20201202014021787](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201202014021787.png)

​															Fig.2 Experiment result[1]

### 	B.Bilinear LSTM

Although traditional LSTM is sensitive to time sequence, it performs not well in real time recognition. In order to optimize the system, bilinear LSTM apply memory module as a part of classifier. Thus the system will get a better result. Here is the framework for Bilinear LSTM[2].

![image-20201203100134495](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201203100134495.png)

​															Fig.3 Framework of Bilinear LSTM[2]

Also, here is the experiment result of bilinear LSTM. IDF1 shows how well the methods perform with proportional relation.

![image-20201203103506624](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201203103506624.png)

​															Fig.4 Experiment result[2]

### 	C.HBU-LSTM

 Hybrid Bidirectional Unidirectional LSTM(HBU-LSTM) is a model which combine unidirectional LSTM and bidirectional LSTM together. The system takes bidirectional LSTM as the first layer of feature collecting[3]. Here is the framework of HBU-LSTM.

![image-20201203103050567](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201203103050567.png)

​														Fig.5 Framework of HBU LSTM[3]

Here is experiment result of HUB-LTSM.

![image-20201203103736161](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201203103736161.png)

​														Fig.6 Framework of HBU LSTM[3]

## 3.Model Design

### 	A.Data set

​	Our data is collected from smart phones which provide many categories of data,such as acceleration and angular speed. Based on APP:phyphox which can collect data of smart phones in certain time sequence. Here is an example of one data set which are acceleration data without g in three axes. The corresponding hand motion is tick and there are 12 times of such motion in collected time sequence.

![image-20201202110209158](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201202110209158.png)

​								Fig.7 Current data set: acceleration data without g,hand motion: tick

We have designed 3 different hand gestures which we use to collect and clean data. The whole practice model is as follows. These three different hand gestures can be used to relate with game events.

![image-20201203104049851](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201203104049851.png)

​															Fig.8 data collection flow chart

### B.LSTM deep learning model and training 

​	They are still developing.



## 4.Current Work

### A.Patent work

Two patents are finished about our topic. Our group finished writing patent which is related to our topic——Development of Asymmetric Competitive Games Based on Bluetooth Technology. Here is a partial view of our patent work.

![image-20201202110703532](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201202110703532.png)

![image-20201203104138903](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201203104138903.png)

​										Fig.9  Partial view of patent work

### B.Literature work

As shown above,we did further more literature work about our adjusted goal. We learned more about hand motion recognition and LSTM network,which is vital for us to build up our own deep learning model.



### C.Tentative draft of essay

We finished tentative draft of essay which is related to our topic. Abstract,introduction, related work parts and part of model design are finished. Here is a partial view of draft of essay.

![image-20201203104230739](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201203104230739.png)

![image-20201203104248188](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201203104248188.png)

​										Fig.10  Partial view of draft of essay



### D.Tentative try on LSTM model

​	We are learning LSTM model and try to build up our own model so that we can start our experiments. We have finished collecting gesture data which are shown above.

![image-20201203163231463](C:\Users\THINKPAD\AppData\Roaming\Typora\typora-user-images\image-20201203163231463.png)

​											Fig.11 partial view of our test codes

## 5.Future Work

First,we will build up our deep learning model as soon as possible since all experiments are dependent on it. This part will be finished before 12.15.

Second,as long as the training finish,we will report our experiment result in our essay. This part will be finished before 12.23.

Third, if time permits, try to collect more groups of data so that training will be more reliable. 





## 6.Reference

[1]B. Wang and M. Hoai, "Predicting Body Movement and Recognizing Actions: An Integrated Framework for Mutual Benefits," 2018 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018), Xi'an, 2018, pp. 341-348, doi: 10.1109/FG.2018.00056.

[2]Kim, C., Li, F., & Rehg, J. M. (2018). Multi-object tracking with neural gating using bilinear lstm. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 200-215).

[3]Ameur, S., Khalifa, A. B., & Bouhlel, M. S. (2020). A novel hybrid bidirectional unidirectional LSTM network for dynamic hand gesture recognition with Leap Motion. Entertainment Computing, 35, 100373.

