import numpy as np
import pandas as pd
import numpy as np
import sys
import gc
import pylab as plt
import os
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, RandomizedLogisticRegression, lasso_stability_path, LassoLarsCV
from sklearn import cross_validation, metrics
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV,RFE,SelectPercentile, f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split as tts




def submitResult(f, user, prob):
    print "Writing results for User ", user
    tripsIndex = range(1,201)
    for t,p in zip(tripsIndex,prob):
        f.write(str(u) + '_' + str(t) + ',' + str(p) + '\n')


def readTripsFromFile(user,debug=False):
    if debug==True:
        print "Reading trips for User ", user
    # list of 2D numpy arrays
    tripsLocs = np.load("drivers/{}/trips.npy".format(user))
    return tripsLocs

def generateFeatures(user_locs, num_trips=500, debug=False):
    '''
    user_locs - list of 2d numpy arrays for all trips    
    num_trips - how many trips to process
    '''

    #Calculating average trip distance, velocity, acceleration
    
    features = []

    tripsIndex = range(1,201)
    for i,trip in enumerate(user_locs):

        if i > num_trips:
            return features

        if debug==True:
            print "Generating features for trip ", i
        velocity = trip[1:,] - trip[:-1,]
        speed = np.sqrt( velocity[:,0]**2 + velocity[:,1]**2 )
        curvature = np.sum(velocity[1:]*velocity[:-1], axis=1)/(speed[1:]*speed[:-1])
        curvature = np.where(np.isnan(curvature), 0 , curvature)
        avg_curvature = np.mean(curvature)
        max_curvature = min(curvature) 
        std_curvature = np.std(curvature)

        if debug==True:
            print trip[:5,]
            print velocity[:5,]
            print speed[:5]
            print curvature[:5]

        avg_speed = np.mean(speed)
        std_speed = np.std(speed)
        max_speed = max(speed)
        min_speed = min(speed)
        trip_distance = np.sum(speed)         

        accel = velocity[1:,] - velocity[:-1]
        abs_accel = np.sqrt( accel[:,0]**2 + accel[:,1]**2 )      
        angle_accel = np.sum(accel*velocity[1:], axis=1)/( abs_accel*speed[1:] )
        angle_accel = np.where(np.isnan(angle_accel), 0 , angle_accel)
        accel = np.where(angle_accel>0, abs_accel, -1.*abs_accel)

        if debug==True:
            print accel[:5]
            print abs_accel[:5]
            print angle_accel[:5]

        avg_accel = np.mean(accel)
        std_accel = np.std(accel)
        max_accel = max(accel)
        min_accel = min(accel)

        f = [avg_speed, std_speed, max_speed, trip_distance, avg_curvature, max_curvature, avg_accel, std_accel, max_accel, min_accel]
        if debug==True:
            print f
        features.append(f)    

    return features


if __name__ == '__main__':

  validation = True
  makeSubmission = True

  users=[]
  probs = []
  for u in os.listdir("./drivers/"):
      users.append(u)
  users = sorted(users)
  #print users[:5]

  for u in users:
      print "Preparing Training set for User ", u
      user_locs = readTripsFromFile(u)
      u_train = generateFeatures(user_locs,debug=False)
      u_train = np.asarray(u_train,dtype=float)
      label_0 = np.ones(u_train.shape[0])      

      pred_probs = np.zeros(200)
      
      num_models = 100

      for nm in range(num_models):

          v_train = []
          if validation == True:
              #print "Preparing Validation set"
              train_users =[u]
              while len(train_users) !=5: #gather trips from 10 other random users
                false_user = users[np.random.randint(len(users))]
                if false_user not in train_users:
                  #print "Chose User ", false_user
                  val_locs = readTripsFromFile(false_user)  
                  v_train.extend( generateFeatures(val_locs,num_trips=25) )
                  train_users.append(false_user)
          v_train = np.asarray(v_train, dtype=float)
          #print "User trip shape and False user trip shape: ", u_train.shape, v_train.shape

          label_1 = np.zeros(v_train.shape[0])      
          if validation == True:
              train = np.vstack((u_train.copy(),v_train.copy()))
          else:
              train = u_train.copy()
          labels = np.concatenate((label_0,label_1))
    
          train -= train.mean(0)
          train = np.nan_to_num(1.*train / train.std(0))

          X_train, X_val, y_train, y_val = tts(train.copy(), labels.copy(), train_size=0.9)
          ''' 
          for reg in [0.001, 0.01, 0.1, 1, 10, 100]:
              clf = LogisticRegression(C=reg,penalty='l2',random_state=0)
              scores = cross_validation.cross_val_score(clf, train, labels, cv=5 )
              print ("Cross Validation(cv=5) scores: reg=%f mean=%f dev=%f" % (reg,scores.mean(),scores.std()) )
          '''
          reg = 0.01
          clf_f = LogisticRegression(C=reg,penalty='l2',random_state=42)
          clf_f.fit(X_train,y_train)
          #print "Classifier:"
          #print clf, clf.get_params()
          #print "Validation set score for best C=: " ,reg, clf_f.score(X_val, y_val)
          pred =  clf_f.predict_proba(train)[:200,1]
          #print pred
          #print  clf_f.predict(train)

          pred_probs += pred

      pred_probs /= num_models
      #print pred_probs
      #print np.where(pred_probs>0.5,1,0)
      probs.append(pred_probs)  
      gc.collect()

  #print len(probs)

  if makeSubmission == True:
      f = open("submit_logistic_ensemble.csv",'w')
      f.write('driver_trip,prob\n')
      for u,pr in zip(users,probs):
          submitResult(f,u,pr) 


