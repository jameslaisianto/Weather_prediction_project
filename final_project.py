import numpy as np
from datetime import datetime
import pandas as pd
from PIL import Image
from pandas import *
import glob
import re
import os
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import pipeline
OUTPUT_TEMPLATE = (
    'Naive Bayes Classifier: {bayes_rgb:.3g}\n'
    'KNN Classifier:      {knn_rgb:.3g}\n'
    'SVM Classifier:      {svm_rgb:.3g}\n'
)


#Read The YVR-Weather Folders
yvr_weather = pd.read_csv('weather-51442-201606.csv', encoding='ISO-8859-1', skiprows=15)
for i in range(7, 13):
    if i < 10:
        data = (pd.read_csv('weather-51442-20160' + str(i) + '.csv', encoding='ISO-8859-1', skiprows=15))
        yvr_weather = yvr_weather.append(data)
        # print(yvr_weather)
    else:
        data = (pd.read_csv('weather-51442-2016' + str(i) + '.csv', encoding='ISO-8859-1', skiprows=15))
        yvr_weather = yvr_weather.append(data)
for i in range(1, 7):
    if i < 10:
        data = (pd.read_csv('weather-51442-20170' + str(i) + '.csv', encoding='ISO-8859-1', skiprows=15))
        yvr_weather = yvr_weather.append(data)


#Convert Datetime Object to NUMERIC Values only
weather_datetime=[]
#Cited: https://stackoverflow.com/questions/39466677/change-datetime-format-in-python
def parse_db_time_string(time_string):
    date = datetime.strptime(time_string.split('.')[0], '%Y-%m-%d %H:%M')
    return datetime.strftime(date, '%Y%m%d%H%M')


#Then change the array into pandas dataframe
for index, row in yvr_weather.iterrows():
    new_string = parse_db_time_string(row['Date/Time'])
    weather_datetime.append(new_string)
weather_datetime = np.asarray(weather_datetime)
yvr_weather['Date/Time'] = weather_datetime


#Drop NA in weather column and dropped other columns with NA
yvr_weather = yvr_weather.dropna(subset=['Weather'])
yvr_weather = yvr_weather.dropna(axis=1, how='any')

def selection(selected):
    if selected == 'Mostly Cloudy':
        return 'Cloudy'
    elif selected == 'Mainly Clear':
        return 'Clear'
    elif selected == 'Rain,Fog':
        return 'Rain'
    elif selected == 'Drizzle,Fog':
        return 'Drizzle'
    elif selected == 'Rain Showers':
        return 'Rain'
    else:
        return selected

yvr_weather['Weather'] = yvr_weather['Weather'].apply(selection)


#Get All Katkam Images in folder
image_file_list= glob.glob('katkam-scaled/*.jpg')
combine_images = np.array([np.array(Image.open(fname)) for fname in image_file_list])
filename = np.array([np.array(fname) for fname in image_file_list])
print(combine_images.shape)


#Using Regular Expression to convert the filename into NUMERIC Values
pic_filename = []
log_pattern = re.compile(
    r'\d\d\d\d\d\d\d\d\d\d\d\d')
for line in filename:
    m = log_pattern.search(line)
    pic_filename.append(m[0])
    if not m:
        raise ValueError('Bad input line: %r' % (line,))

pic_filename = np.asarray(pic_filename)


#Flatten matrix array into 1D and turn it into a list
flatten_matrix = combine_images.reshape(5046,-1)
print(flatten_matrix)


#Create a filename dataframe and then append the image matrix together
flat_file = pd.DataFrame(flatten_matrix)
flat_file['Date/Time'] = pic_filename
print(flat_file)
# # pd.set_option('display.max_columns', 500)
# print (yvr_weather)


#To get only the weather and not the other columns from weather data
# weather_only = pd.DataFrame(yvr_weather['Date/Time'])
# weather_only['Weather'] = yvr_weather['Weather']
# print('Weather',weather_only)

#Convert Time column to 2-digt value
time=[]
#Cited: https://stackoverflow.com/questions/39466677/change-datetime-format-in-python
def parse_db_time_string(time_string):
    date = datetime.strptime(time_string.split('.')[0], '%H:%M')
    return datetime.strftime(date, '%H%M')


#Then change the array into pandas dataframe
for index, row in yvr_weather.iterrows():
    new_string = parse_db_time_string(row['Time'])
    time.append(new_string)
time = np.asarray(time)
yvr_weather['Time'] = time

print(yvr_weather)




#Join both the 2D array(filename and matrix) with the yvr-weather data
append_both = yvr_weather.merge(flat_file, on='Date/Time')
# append_both = weather_only.merge(flat_file, on='Date/Time')


# #Learning Part
y = append_both['Weather']

X = append_both
del append_both['Weather'], append_both['Data Quality'], append_both['Date/Time']
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)
y_train = y_train.ravel()

# 1st Part
# naive Bayes Classifier
bayes_rgb_model = naive_bayes.GaussianNB()
bayes_rgb_model.fit(X_train, y_train)

# KNN Classifier
knn_rgb_model = KNeighborsClassifier(n_neighbors= 30)
knn_rgb_model.fit(X_train, y_train)

#SVM Classifier
svc_rgb_model = SVC(kernel='linear', C=5.0)
svc_rgb_model.fit(X_train, y_train)


print(OUTPUT_TEMPLATE.format(
    bayes_rgb=bayes_rgb_model.score(X_test, y_test),

    knn_rgb=knn_rgb_model.score(X_test, y_test),

    svm_rgb=svc_rgb_model.score(X_test, y_test),

))
