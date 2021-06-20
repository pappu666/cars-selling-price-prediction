from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#from sklearn.model_selection import RandomizedSearchCV

def home(request):
    return render(request,'index.html')

def result(request):
    df=pd.read_csv('car data.csv')
    final_dataset=df[[ 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven','Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
    final_dataset['Current_Year'] = 2020
    final_dataset['Total_Year']=final_dataset['Current_Year']-final_dataset['Year']
    final_dataset.drop(['Year'],axis=1,inplace=True)
    final_dataset.drop(['Current_Year'],axis=1,inplace=True)
    final_dataset=pd.get_dummies(final_dataset,drop_first=True)
    x=final_dataset.iloc[:,1:]
    y=final_dataset.iloc[:,0]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    
    rr=RandomForestRegressor()
    #n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
    #max_features=['auto','sqrt']
    #max_depth=[int(x) for x in np.linspace(5,30,num=6)]
    #min_samples_split=[2,5,10,15,100]
    #min_samples_leaf=[1,2,5,10]
    #random_grid={'n_estimators':n_estimators,
            #'max_features':max_features,
            #'max_depth':max_depth,
            #'min_samples_split':min_samples_split,
            #'min_samples_leaf':min_samples_leaf}
    #r=RandomizedSearchCV(estimator=rr,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)
    rr.fit(x_train,y_train)
    if request.method == 'POST':
        Year=int(request.POST['Year'])
        Total_Year = 2020 - Year
        
        Present_Price=float(request.POST['Present_Price'])
        Kms_Driven=int(request.POST['Kms_Driven'])
        Owner=int(request.POST['Owner'])
        Fuel_Type_Petrol=request.POST['Fuel_Type_Petrol']
        if(Fuel_Type_Petrol=='Petrol'):
            Fuel_Type_Petrol=1
            Fuel_Type_Diesel=0
        elif(Fuel_Type_Petrol=='Diesel'):
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=1
        else:
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=0
        

        Seller_Type_Individual=request.POST['Seller_Type_Individual']
        if(Seller_Type_Individual=='Individual'):
            Seller_Type_Individual=1
        else:
            Seller_Type_Individual=0
        Transmission_Manual=request.POST['Transmission_Manual']
        if(Transmission_Manual=='Manual'):
            Transmission_Manual=1
        else:
            Transmission_Manual=0
        pappu=rr.predict([[Total_Year,Present_Price,Kms_Driven,Owner,Fuel_Type_Petrol,Fuel_Type_Diesel,Seller_Type_Individual,Transmission_Manual]])
        return render(request,'index.html',{'prediction_text':pappu})
    else:
        return render(request,'index.html')