import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
dfFlight=pd.read_csv('flight_data.csv')
#print(dfFlight)
#print(reading_flight['flight'][1])
#checks the number of missing values in each column as isnull() 
#print(dfFlight.shape)
df_compress=dfFlight.apply(lambda x: sum(x.isnull()),axis=0)
#print(df_compress)
#print(df_compress.shape)
#remove_features=reading_flight.drop()

#for i in reading_flight:
#   print(i)
# Removing rows with missing values    
dfFlight_duplicate= dfFlight[~pd.isnull(dfFlight).any(axis=1)]#displayed data only with team != NaN 
#print(dfFlight_duplicate.shape)
#dfFlight_duplicate= dfFlight[~pd.isnull(dfFlight).any(axis=0)]
#print(dfFlight_duplicate.shape)


#The next few blocks of code find the busiest routes
#and arrange them in order of their traffics or how busy they are.
lsOrig = dfFlight.origin.unique().tolist()
lsDest = dfFlight.dest.unique().tolist()
#print(lsOrig)
#print(lsDest)
lsRouteOr = []
lsRouteDe = []
lsRouteOrName = []
lsRouteDeName = []
lsRouteFq = []
for idOr in lsOrig:
    for idDe in lsDest:        
        if idOr != idDe: 
            Freq = dfFlight.loc[(dfFlight['origin'] == idOr) & (dfFlight['dest'] == idDe)].shape[0]
            #print(Freq)
            if Freq > 0:
                lsRouteOr.append(idOr)
                lsRouteDe.append(idDe)
                lsRouteFq.append(Freq)
#print(lsRouteOr)
#print(lsRouteDe)
#print(lsRouteFq)
# Create dataframe
Route = {'origin': lsRouteOr,'dest': lsRouteDe,'RouteFreq': lsRouteFq}
dfRoute = pd.DataFrame(Route, columns = ['origin','dest','RouteFreq'])
dfRoute = dfRoute.sort_values(by='RouteFreq', ascending=False)
dfRoute = dfRoute.reset_index(drop=True)

#print(dfRoute.shape)
dfRoute.head()
#print(dfRoute)
origin_cat=dfRoute['origin']
dest_cat=dfRoute['dest']
#print(origin_cat)
#creating id for both origin_place and dest_place by label encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
origin_id=labelencoder_X.fit_transform(origin_cat)
dest_id=labelencoder_X.fit_transform(dest_cat)
#print(origin_id)
#print(dest_id)
new_df=dfRoute.assign(origin_id=origin_id,dest_id=dest_id)
#new_df=dfRoute.columns['origin_id']=origin_id
#new_df=dfRoute['origin_id']=origin_id
#print(new_df)
#pd.DataFrame(df)
#here visvualize few data with seaborn out off 224 records only with few data
#and outcome record we can check by origin_id_dest_id(0_11)=value is routeFreq
full=new_df+dfFlight_duplicate
print(full)
def dfRouteFtn(dfRoute10, itrMax = 10):
    lsOrDe = []
    lsOr_De = []
    lsDir = []
    lsRouteFq = []
    itr =  0
    for OrId, DeId, Freqx in zip(dfRoute10['origin_id'], dfRoute10['dest_id'], dfRoute10['RouteFreq']):
         #print(OrId)
        # print(DeId)
         #print(Freqx)
         #print(dfRoute10.RouteFreq.to_string(index=False))
         #print(dfRoute10.RouteFreq.to_string(index=True))
         #print(dfRoute10.loc['1'].RouteFreq.to_string())
         #print((dfRoute10['origin_id'] == DeId) | (dfRoute10['dest_id'] == OrId))
         Freqy = dfRoute10.loc[(dfRoute10['origin_id'] == DeId) | (dfRoute10['dest_id'] == OrId)].RouteFreq.to_string(index=False)
         #print(Freqy.isnumeric())
         #print(Freqy)
         if (str(OrId) + "_" + str(DeId)) not in lsOrDe and Freqy.isnumeric():
            # print("rakesh")
             lsOr_De.append(str(OrId) + "_" + str(DeId)), lsRouteFq.append(Freqx), lsDir.append("Or_De")
             #print(lsOr_De)
             lsOr_De.append(str(OrId) + "_" + str(DeId)), lsRouteFq.append(Freqy), lsDir.append("De_Or")
           #  print(lsOr_De)
             lsOrDe.append(str(DeId) + "_" + str(OrId))
            # print(lsOrDe)
             itr =itr+1
             if itr == itrMax: break
    
    # Create dataframe
    Routx = {'Origin_Destination': lsOr_De,'RouteFreq': lsRouteFq, 'RouteDirection':lsDir }
    dfRoutx = pd.DataFrame(Routx, columns = ['Origin_Destination','RouteFreq', 'RouteDirection'])
    return dfRoutx
data = dfRouteFtn(new_df)
data.RouteFreq = pd.to_numeric(data.RouteFreq, errors='coerce')
BarRects = sns.barplot(x=data.Origin_Destination, y=data.RouteFreq, palette=sns.color_palette("cubehelix"),ci=None, hue=data.RouteDirection)

plt.xticks(rotation=60) #BarPlotInd+.5, BarLabels)
plt.xlabel('Origin-Destination Airport Id')
plt.ylabel('Route Flight Frequency')
#plt.ylim([0,1])
plt.title('Ten Busiet Flight Routes in US in April-October 2013')

plt.show()
#print(dfFlight['origin'])
origin_cat=dfFlight_duplicate['origin']
dest_cat=dfFlight_duplicate['dest']
#print(origin_cat)
#print(dest_cat)
#creating id for both origin_place and dest_place by label encoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
origin_id2=labelencoder_X.fit_transform(origin_cat)
dest_id2=labelencoder_X.fit_transform(dest_cat)
#return1=count(origin_id2)
#return1=origin_id2.count()
#print(return1)
#print(origin_id2)
#print(dest_id2)
#print(origin_id)
#print(dest_id)
created=dfFlight_duplicate.assign(origin_id=origin_id2,dest_id=dest_id2)
print(len(created))
##print(created)

# This function gives the airport ID of the busiest airport in a particular city.
def MaxFreqAirportId(city):
    if city in new_df["origin"]:
        if city is EWR:
            return 0
        elif city is JFK:
            return 1
        else:
            return 2
    else:
        return 35


# This function depicts the reationsship between the busiest airports in two different cities.
# This function depicts the reationsship between the busiest airports in two different cities.
def FlightRoutePlot(dfx,DepAirportCity,ArrAirportCity,xFea,yFea):
    #print(xFea)
    #print(dfx)
    #print(DepAirportCity)
    
    
    #dfx1=dfFlight_duplicate# Departure and arrival airport cities and the feature of interest     
    selFea = ['month', 'day', 'hour', 'carrier', 'arr_delay']
    #dfx1=dfFlight_duplicate
    dfx2 = dfx.loc[(dfx['origin_id'] == MaxFreqAirportId(DepAirportCity)) | (dfx['dest_id'] == MaxFreqAirportId(ArrAirportCity)), selFea]# Only get flights for our flight route    
    
    # For flights that arrived earlier than scheduled time (i.e. with negative delay time) are converted to zero 
    # since we are only after delay time.
    #print(dfx2)
    dfx2['arr_delay'] = [0 if x < 0 else x for x in dfx2['arr_delay']]    
    # Add a column for months
    dfx2['month'] = ['Jan' if x in [1] else
                                 ('Feb' if x in [2] else 
                                 ('Mar' if x in [3] else
                                 ('Apr' if x in [4] else
                                 ('May' if x in [5] else 
                                 ('Jun' if x in [6] else
                                 ('Jul' if x in [7] else
                                 ('Aug' if x in [8] else 
                                 ('Sep' if x in [9] else
                                 ('Oct' if x in [10] else
                                 ('Nov' if x in [11] else 
                                 'Dec' )))))))))) for x in dfx2['month']] 
    
    BarRects = sns.barplot(x=dfx2[xFea], y=dfx2[yFea], palette=sns.color_palette("cubehelix"),ci=None, estimator=np.mean)

    plt.xticks(rotation=60) 
    plt.xlabel(xFea)
    plt.ylabel('Route Flight Frequency')
    plt.title('Flights from ' + str(DepAirportCity) + ' to ' + str(ArrAirportCity))
    plt.show()
                                                                                       
FlightRoutePlot(created,'EWR','FLL','carrier', 'arr_delay')
FlightRoutePlot(created,'EWR','FLL','month', 'arr_delay')

#model_Prediction
#one question rised is,is that class label as negative values here why i got that error
#is i thought of taking arr_delay as my atribute but there is negative value.
#
#Model_Prediction
#import model libraries
import scipy.stats as st
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#Import module for saving scikit-learn models - Joblib is an alternative to Python's pickle package
from sklearn.externals import joblib
Y=dfFlight_duplicate.arr_delay
print(Y)
#we can't drop multiple as we done for single drop os attributeX = dfFlight_duplicate.drop(['dep_delay'],axis=1)

drop_list=['arr_delay','sched_dep_time','dep_delay','sched_arr_time','flight','tailnum','air_time','time_hour','distance','hour','minute']
#X=dfFlight_duplicate.drop(['sched_dep_time'],['dep_delay'],['sched_arr_time'],['flight'],['tailnum'],['air_time'],['time_hour'],['distance'], axis=1)
#such type drop of all at time drop can't be expected only df.drop(['flight'],axis=1)

#X=dfFlight_duplicate.drop(['sched_dep_time'],['dep_delay'],['sched_arr_time'],['flight'],['tailnum'],['air_time'],['time_hour'],['distance'])
X = dfFlight_duplicate.drop(drop_list,axis=1)
print(X)
"""X = dfFlight_duplicate.drop(['sched_dep_time'],axis=1)
#print(X)
X = dfFlight_duplicate.drop(['dep_delay'],axis=1)
X = dfFlight_duplicate.drop(['sched_arr_time'],axis=1)
X = dfFlight_duplicate.drop(['flight'],axis=1)
X = dfFlight_duplicate.drop(['tailnum'],axis=1)
X = dfFlight_duplicate.drop(['air_time'],axis=1)
X = dfFlight_duplicate.drop(['time_hour'],axis=1)
X = dfFlight_duplicate.drop(['distance'],axis=1)
print(X)
"""
CategLs = ['month', 'year', 'day', 'carrier', 'origin', 'dest', 'dep_time', 'arr_time'] # Categorical features
for fea in X[CategLs]: # Loop through all columns in the dataframe
    X[fea] = pd.Categorical(X[fea]).codes # Convert to categorical features
print(X)
#Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=123)
one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive
}

xgbreg = XGBRegressor(nthread=-1)
rsCV = RandomizedSearchCV(xgbreg, params, n_jobs=1)  
rsCV.fit(X_train, Y_train)
rsCV.best_params_, rsCV.best_score_

clf = XGBRegressor(**rsCV.best_params_)
print(clf)
clf.fit(X_train, Y_train)
#from sklearn.metrics
import sklearn.metrics
result=clf.predict(X_test)
print("MAE: %.4f" % mean_absolute_error(Y_test, clf.predict(X_test)))
print("MSE: %.4f" % mean_squared_error(Y_test,clf.predict(X_test)))
#score = metrics.mean_squared_error(result, Y_test)
#print(score)


