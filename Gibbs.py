import math
import numpy


OFmatrix = numpy.matrix('0,0,0,0,0,0,0,0,0,0;0,0,0,0,0,0,0,0,0,0;1,1,1,1,1,1,1,1,1,1;1,1,1,1,1,1,1,1,1,1')
data = OFmatrix

gamma = 0.5
xa1=xa0=0.5
alph1 = alph0 = 0.5
data = numpy.loadtxt('testoutput.txt',dtype=int)





## ================================The home base=====================    
    
class GibbsSampler:

    def __init__(self,datamatrix):
        self.datamatrix = datamatrix       
        self.numobj = datamatrix.shape[0]
        self.numfeatures = datamatrix.shape[1]
        self.statematrix = numpy.zeros((1,self.numobj))
        self.table = numpy.zeros((self.numobj,self.numobj))

    ## This is the actual Gibbs Sampler. It runs the algorithm and "builds" the matrix of
    ## observed category vectors
    def Gibbyribby(self,gamma,aph1,aph0,z_init,trials):
        z0 = catinit(self.datamatrix,z_init)
        
        for trial in range(trials):
            placeholder = Zmaker(self.datamatrix,z0,gamma,aph1,aph0)
            
            ## Only recorders every 100
            if( trial % 20 == 0):
    ##        print(placeholder)
                self.statematrix = numpy.append(self.statematrix,placeholder,axis=0)

        self.statematrix = self.statematrix.astype(int)


    ## This is the function that predicts the most likely categorization, using the
    ## Class sampler's data matrix
    def estimator(self):
        table = numpy.zeros((self.statematrix.shape[1],self.statematrix.shape[1]))
                            
        for j in range(self.statematrix.shape[1]):
            temp = self.statematrix[:,j]

            for i in range(self.statematrix.shape[1]):
                table[j,i] = (temp == i).sum()

        estimate = numpy.zeros((1,self.statematrix.shape[1]))
        for k in range(self.statematrix.shape[1]):
            estimate[0,k] = numpy.argmax(table[k,:])
        estimate = estimate.astype(int)
        print(estimate)
    
                
## ================================Z maker================================
            
##What will make the new statevector and append it to old list
def Zmaker(datar,z_crnt,gamma,a1,a0):
    data = datar.transpose()

    ## Random integer among 0 -> (#number of obj)
    ## This is the element we will change
    elmt = numpy.random.random_integers(0,z_crnt.shape[1]-1)

    ## number of categories in z_crnt
    k = numpy.amax(z_crnt)

    ## The matrix containing the probabilities of
    ## moving to the next state
    QQ = Qmatrix_calc(data,z_crnt,k,elmt,a1,a0)

    ## New possiblilities
    newpos = QQ.shape[1]

    ## Picking the new value randomly using QQ
    newvaltemp = numpy.random.choice(numpy.arange(newpos),1,p=[QQ[0,i] for i in range(newpos)])
    
    ## the value needs an extra +1 to prvent 0
    newval = newvaltemp[0] + 1

    
    ## copy
    z_new = z_crnt
    ## change the value
    z_new[0,elmt] = newval

    ## reduce the values to prevent things like
    ## [1,1,3,3,5,5]
    z_new = reducer(z_new)

    return z_new

##==================================Vector Reducer========================
def reducer(z):
    ## this function will reduce the values of our catagory matrix. For example, the category matrix [2,3,2,3] is
    ## no different than [1,2,1,2] and our functions will work "bottom up" so we need this reduction.
    ##print(numpy.amax(z))
    for check in range(numpy.amax(z)):
        
        use = check + 1
        if( (z == use).sum() == 0):
            for i in range(z.shape[1]):
                if (z[0,i] > use):
                    z[0,i] = z[0,i]-1

    return z
## =================================Q matrix==============================

## Calculating the Q matrix which is the matrix that will contain the
    ## distribution for moving to the next state.
def Qmatrix_calc(data,z_crnt,k,elmt,a1,a0):

    ## We use this to find out how many categories are occupied now
    ## then we use that number + 1 to allow for the possible new
    ## categories for our new c vector
    ni_mat = ni_matrixmaker(z_crnt,k)

    eltval = z_crnt[0,elmt]

    ## If highest category is more than the number of realized:
    
   
    ## If all categories are realized, don't allow for new one
    if (ni_mat.shape[1] == data.shape[0]):
        pos_cat = ni_mat.shape[1]
    ## If the object in question is already in a category by itself, we don't add the new category as that
    ## would be the same as staying alone
    elif ( (z_crnt == eltval).sum() == 0):
        pos_cat = ni_mat.shape[1]
    ## Finally if not then we allow for the possibility of a new category    
    else:
        pos_cat = ni_mat.shape[1] + 1

    ## initializing the distribution matrix
    Q = numpy.zeros((1,pos_cat))

    
    ## iterating over all columns
    for i in range(Q.shape[1]):

        ## the new statevector in the making (copied from old)
        new_z = z_crnt
        ## changning the category (+1)
        new_z[0,elmt] = i+1

        ##when we run the pDGca, it requires a category count
        ##We need to just tell it how many categories there are
        ## being realized, so we check for the max value in
        ## our state vector.
        cat_realized = numpy.amax(new_z)

        ##The i-th entry in our Q matrix is the probability of moving to the (i+1)th category
        ##Note the output of pDGca is logarithmic so we revert to standard probability
        Q[0,i] =  pDGca(data,new_z,cat_realized,0.5,0.5,0.5)

    Q[:] = [x - numpy.amax(Q) for x in Q] 
    Q[:] = [math.e ** x for x in Q]
    Q[:] = [x / Q.sum() for x in Q]

    return Q        
#=====================================================================================================
## This function makes a "closed" form of the beta function for us, in that
## this evalutates B(ti + alph0,fi + alph1),B(alph0,alph1) without using gamma functions
## Necessary? Maybe not, but oh well
def Bsimp(ti,fi,xa1,xa0):
    top1 = 1.0

    for i in range(ti):
        top1 = top1 * (xa1 + i)

    top2= 1.0
    for i in range(fi):
        top2 = top2 * (xa0 + i)

    bot1 = 1.0
    for i in range(ti+fi):
        bot1 = bot1 * (xa0 + xa1 + i)

    return top1 * top2 / bot1

##=============================================Logbetabernoulli==============================================

def logBetaBernoulli(j,c,k,xa1,xa0):
##    if (j.shape[1] != c.shape[1]):
##        print('Shiver me timbers, the data vector and category vector are not the same size')
##    
    

    ## Getting N even if it isn't explicitly given
##    print (j)
    numobj = len(j) 

    ## Our dummy variable
    pDjGca = 0

    ##Looping through the various categories
    ##Setting what category to check
    for i in range (k):
        cat = i+1

        ## Initializing ti fi
        ## We reinitialize for each category
        ti = 0
        fi = 0

       
        ## Now to check if an object is in category "i" and has the feature for
        ## j
        for dex in range(numobj):
            if ( j[dex] == 1 and c[0,dex] == cat):
                ti = ti + 1

        ## Getting fi 
        for dex in range(numobj):
            if ( j[dex] == 0 and c[0,dex] == cat):
                fi = fi + 1

        ## This will multiply the total proability we seek by a single prodcut
        ## term from equation (3) in the problem set. This is done with the
        ## above-defined function "Bsimp" for "Beta simplifier"
        pDjGca = pDjGca + math.log(Bsimp(ti,fi,xa1,xa0),math.e)
##        pDGCA = pDGCA + math.log(Bsimp(ti,fi,xa1,xa0),math.e)

##        print('Category:', cat)
##        print('ti',ti,'fi',fi)
##        print('Log probability of data_j given c,alpha: ',pDjGca)

    return pDjGca


##=====================================Category Counter Matrix======================================

## This function is used to tally the number of objects in each
    ## category by taking the category matrix as input
    ## as well as the number of categories,
    ## then outputting a 1xk matrix with n_i as the entries.
    ## It is called in the logCrp function.

## BIG NOTE : This function will technically 'over count' categories
## Given a z = [1,1,4,4], this function will create a "count matrix" and should return
## [2,0,0,2] This is okay because in the Bsimp function which will use this has been set to say that
## (0 - 1)! = 1, and so that our LogCrp function is unaffected.

def ni_matrixmaker(catmatrix,k):
    ## need this to set up our counter, # of columns
    cols = catmatrix.shape[1]

    ## the n_i matrix 
    ni_matrix = numpy.zeros((1,k))

    ## our counter to sweep the catmatrix and record
    for i in range(k):
        ni_count = 0
        for j in range(cols):
            if (catmatrix[0,j] == (i + 1)):
                ni_count = ni_count + 1
        ni_matrix[0,i] = ni_count

    return ni_matrix

##===================================LogCrp============================

def logCrp(gam,k,catmatrix):
    fac = math.factorial
    ##Base value
    top = 1.0
    ## Calling the above function to make our ni matrix
    ni_matrix = ni_matrixmaker(catmatrix,k)
    ##Set this up for our interator
    cols = ni_matrix.shape[1]

    ## Number of objects
    numobj = catmatrix.shape[1]

    ##computing the numerator. Notice we use "cols" because that is
    ## the number of n_i we have
    for i in range(cols):
        if (ni_matrix[0,i] == 0):
            top = top * 1
        else:
            top = top * fac(ni_matrix[0,i]-1)

    ##computing the denomionator. this time we use "numobj" since we needed
        ## N + gamma before
    bot = 1.0
    for i in range(numobj):
        bot = bot * (gam + i)
        
    ##print(numobj,cols)
    ##print(top,bot,gam**k)
    return math.log(top * gam**(k) / bot,math.e)

def logPdGc(datamatrix,catmatrix,k,alpha1,alpha0):
    #We will go row by row through the datamatrix

    #End result initializer:
    logPdGc = 0.0

    lop = datamatrix.shape[0]
    #initializing the iterator for row by row
    for feat in range(lop):
        # Extracting different rows

        j = datamatrix[feat,:]


        #calling the logbetabernoulli function for each row
        pdJ_gca = logBetaBernoulli(j,catmatrix,k,alpha1,alpha0)
        
        # the output of LBB is in log form naturally so we add to our complete probability.
        logPdGc = logPdGc + pdJ_gca
        ##print('Current running sum of pDGca = ',pdgca)
        
    return logPdGc

## ===============================P(c|d,a,gam)================================================
## ===============================Unnormalized================================================
## This is the over-arching function that will answer 1-d for us.

def pDGca(datmatrix,catmatrix,k,gamma,alpha1,alpha0):
    logPdGca = 0.0     

    ## This gives the P(d|c,a)
    logPdGca = logPdGca + logPdGc(datmatrix,catmatrix,k,alpha1,alpha0)

    
    #We now call the P(c | gamma) function a.ka logCrp:
    ## note that the ni_matrixmaker function is included in the logCrp function
    crp = logCrp(gamma,k,catmatrix)

    #add our logCrp output to Logbetabernoulli
    logPdGca = logPdGca + crp

    return logPdGca

class catvector:
    def __init__(self,prev):
        self.Q = numpy.zeros((1))
        self.list = prev
        

##==========================Category Initializer=====================
def catinit(data,arg):
    ##how big to make the category vector
    numobj = data.shape[0]

    ##initialize
    z0 = numpy.zeros((1,numobj))


    ## WAY 1 : randomly initialize
    if ( arg == 1):
        for i in range(numobj):
            z0[0,i]=numpy.random.random_integers(1,numobj)

    ## WAY 2 : ALL SAME
    if ( arg == 2):
        z0[:] = [1 for x in z0]

    ## WAY 3 : ALL DIFFERENT
    if (arg == 3):
        for i in range(numobj):
            z0[0,i] = i+1
            
    ##cast all entries as int
    z0 = z0.astype(int)
    
    #run the reducer to prevent island categories
    for i in range(10):
        z0 = reducer(z0)

    #print(z0)
    return z0




