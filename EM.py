#Declare and create finctions 
from pickle import TRUE
from xml.dom.minidom import TypeInfo
import numpy as np
import pandas as pd
import math
import sklearn.cluster
import sklearn.metrics

# reading in data
# Data =  pd.read_csv('pr3.data', sep=",")
Data =  pd.read_csv('pr.data', sep=",")




#################################################
#Name: EStep
#Description: Performs the E-step 
#inputs: k (k-means clustering), data,  sigma, mu values 
#outs: array of expected values for hidden variables
##################################################
def Estep(k, D, s, mu):
     i=0 #index variable
     etable= {}
   
     length = D.shape[0]

     for l in range(length):  
       
        #for nameing purposes for dict
        datapoint = str(i)

        #no matter k value need at least the first two
        numerator1 = math.exp(((-1)/(2*s))* math.sqrt((D['X'].iloc[i]-mu['X'].iloc[0])**2 + (D['Y'].iloc[i]-mu['Y'].iloc[0])**2))
        numerator2 = math.exp(((-1)/(2*s))* math.sqrt((D['X'].iloc[i]-mu['X'].iloc[1])**2 + (D['Y'].iloc[i]-mu['Y'].iloc[1])**2))
        denominator = numerator1 + numerator2

        #if k is 3 or 4 then need a third mu calculation
        if (k == 3 or k ==4): 
             numerator3 = math.exp(((-1)/(2*s))* math.sqrt((D['X'].iloc[i]-mu['X'].iloc[2])**2 + (D['Y'].iloc[i]-mu['Y'].iloc[2])**2))
             denominator = denominator + numerator3
            
            #if k is 4 then need a fourth mu calculation
             if (k == 4): 
                numerator4 = math.exp(((-1)/(2*s))* math.sqrt((D['X'].iloc[i]-mu['X'].iloc[3])**2 + (D['Y'].iloc[i]-mu['Y'].iloc[3])**2))
                denominator = denominator + numerator4
                value4 = numerator4/denominator
                keyname4 = datapoint + " mu4"
                etable.update({keyname4 : value4})

             value3 = numerator3/denominator
             keyname3 = datapoint + " mu3"
             etable.update({keyname3 : value3})
        
        
        value1 = numerator1/denominator
        value2 = numerator2/denominator

        
        keyname1 = datapoint + " mu1"
        keyname2 = datapoint + " mu2"
        etable.update({keyname2 : value2})
        etable.update({keyname1 : value1})
    
        i= i+1
        
     return etable 
    

       
#################################################
#Name: MStep
#Description: Performs the M-step 
#inputs: EStep array, dataset, k (k-means clustering)
#outs: updated mu values (sample means)
##################################################
def Mstep(etable, D, k):
   #intalize variables
    datapoint = 0
    numerator1x = 0
    numerator2x = 0
    numerator3x = 0
    numerator4x = 0
    numerator1y = 0
    numerator2y = 0
    numerator3y = 0
    numerator4y = 0
    
    denominator1 = 0
    denominator2 = 0
    denominator3 = 0
    denominator4 = 0
    
    length = D.shape[0]

    for l in range(length):   
        
        
        etableKey1 = str(datapoint) + " mu1"
        etableKey2 = str(datapoint) + " mu2"
        etableKey3 = str(datapoint) + " mu3"
        etableKey4 = str(datapoint) + " mu4"

        #no matter k mu1 and mu2 will exist 
        numerator1x = numerator1x + (D['X'].iloc[datapoint] * etable[etableKey1])
        numerator1y = numerator1y + (D['Y'].iloc[datapoint] * etable[etableKey1])
        denominator1 = denominator1 + etable[etableKey1] 
         
        numerator2x = numerator2x + (D['X'].iloc[datapoint] * etable[etableKey2])
        numerator2y = numerator2y + (D['Y'].iloc[datapoint] * etable[etableKey2])
        denominator2 = denominator2 + etable[etableKey2] 
       
        # if k is 3 or  4 we need mu3 to be calculated
        if (k == 3 or k == 4) : 
            numerator3x = numerator3x + (D['X'].iloc[datapoint] * etable[etableKey3] )
            numerator3y = numerator3y + (D['Y'].iloc[datapoint] * etable[etableKey3] )
            denominator3 = denominator3 + etable[etableKey3] 
            
        # if k is  4 we need mu4 to be calculated
        if (k == 4) : 
            numerator4x = numerator4x + (D['X'].iloc[datapoint] * etable[etableKey4])
            numerator4y = numerator4y + (D['Y'].iloc[datapoint] * etable[etableKey4])
            denominator4 = denominator4 + etable[etableKey4] 

        datapoint = datapoint +1 #increase datapoint to look at next data point in the data frame
    
    #calculated needed numerators and denomiators
    #now calculate updated mu
    
    #no matter k need mu1 and mu2 calculations
    updatedm1x = numerator1x / denominator1 
    updatedm1y = numerator1y / denominator1
    updatedm2x = numerator2x / denominator2
    updatedm2y = numerator2y / denominator2
    
    #check to see if need to calcuate mu3 and calculate as necessary
    if (k ==3 or k ==4): 
        updatedm3x = numerator3x / denominator3
        updatedm3y = numerator3y / denominator3
        
        #check to see if need to calcuate mu4 and calculate as necessary
        if (k ==4): 
            updatedm4x = numerator4x / denominator4
            updatedm4y = numerator4y / denominator4
            #now update mu if/for k =4 
            mu = {'X': [updatedm1x, updatedm2x, updatedm3x, updatedm4x ], 'Y': [updatedm1y, updatedm2y, updatedm3y, updatedm4y]}
        else: 
            #now update mu if/for k =3
            mu = {'X': [updatedm1x, updatedm2x, updatedm3x ], 'Y': [updatedm1y, updatedm2y, updatedm3y]}
    else: 
        #now update mu if/for k =2
        mu = {'X': [updatedm1x, updatedm2x ], 'Y': [updatedm1y, updatedm2y]}
    
    
    #now return updated mu values 
    updatedmu = pd.DataFrame(data=mu)
    return updatedmu

    
        
################
##### MAIN ##### 
################

#declare the different k and sigma values wanted to be used
kVals = [2, 3, 4]
sigmaVals = [.5 , 1,  2]
runagainBool = True

#run through each k and sigma combination
for i in range(5):
    for k in kVals: 
        for s in sigmaVals: 

            #reset while loop bool
            runagainBool = True
            #assign intial hypothesis
            if (k==2): 
                arrayindex =  np.random.choice(range(Data.shape[0]), replace = False, size = k)
                mu = Data.loc[[arrayindex[0],arrayindex[1]]]
            elif (k==3): 
                arrayindex =  np.random.choice(range(Data.shape[0]), replace = False, size = k)
                mu = Data.loc[[arrayindex[0],arrayindex[1], arrayindex[2]]]
            else: 
                arrayindex =  np.random.choice(range(Data.shape[0]), replace = False, size = k)
                mu = Data.loc[[arrayindex[0],arrayindex[1], arrayindex[2],  arrayindex[3]]]
            
            muStart = mu
            #if the change in the avergae mean is greater than .05 continue
            while(runagainBool == True):
                #call E-step 
                etable = Estep(k, Data, s, mu)

                # call M-step
                muupdated = Mstep(etable, Data, k)

                #calculate change and see if need to run again
                difference = abs(muupdated.subtract(mu))
                runagain = (difference > .005).any()
                runagainlist = runagain.tolist()
                runagainBool = any(x  == True for x in runagainlist)
                
                #reset mu to be muupdtaed 
                mu = muupdated

            
            print("Initalized mean values:" )
            print(muStart)
            print("For " , k , "-means with sigma = ", s, " the mean values are : " )
            print(mu)
            

            # Davies-Bouldin algorithm 
            kmModel = sklearn.cluster.KMeans(n_clusters=k).fit(Data)
            print("For " , k , "-means with sigma = ", s, " the Davies-Bouldin algorithm score : ")
            print(sklearn.metrics.davies_bouldin_score(Data, kmModel.labels_))
            print()








