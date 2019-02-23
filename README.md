# Recognition-Hand-Gestures

Application of predictive linear and non-linear models to predict the wrist movements (like flicking hand right or left) from realistic EMGs. In other words, run EMG data into predictive model that builds a mapping function between EMG and arm movement, and also visualize the data. The models used here are Linear Discriminate Analysis and Neural Network. The predictive capabilities of the learnt model are analyzed in a held out test data. The model was tested based on different gestures, different segment size and different samples per task. The accuracy was improved when the data was split into shorter segments.

Data is obtained from Myo Armband https://support.getmyo.com/hc/en-us/articles/203398347-Getting-started-with-your-Myo-armband
