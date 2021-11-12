##################################################################################################################
#define a function to be called on later

def sumN(x):  
    j=1
    z=0
    for j in range(0,x+1):
        z=j+z
        j=j+1
    return z

##################################################################################################################
#define a function for odd sum

def sumoddN(x): 
    i=1
    y=0
    while i<=2*x:
        y=i+y
        i=i+2
    return y

##################################################################################################################
#define a factorial function

def factorial(x): 
    i=1
    j=1
    while j<=x:
        i=i*j
        j+=1
    return i

##################################################################################################################
##################################################################################################################
##################################################################################################################
#define function which displays a matrix in standard form instead of like an array

def mDisplay(A):
    for i in range(len(A)):
        for j in range(len(A[i])):
            print((A[i][j]), end = " ")
        print()

##################################################################################################################
#define a function for finding the transpose of a matrix

def mTranspose(a):
    n = len(a)
    aT = [[0 for i in range(n)] for j in range(n)]
    for i in range(len(a)):
        for j in range(len(a[0])):
            aT[j][i] = a[i][j]
    return aT

##################################################################################################################
#define a function which displays the product of of two matrices

def mMultiply(A,B):
    X=[[0 for i in range(len(B[0]))] for j in range(len(A))] #creating the variable matrix
    for i in range(len(A)): #rows of first matrix
        for j in range(len(B[0])): #columns of second matrix, all the rows have same no. of coulmns as this is a matrix so we can take the first row
            for k in range(len(A)):
                X[i][j] = X[i][j] + (A[i][k] * B[k][j]) #operating on individual elements and inputting them in X
    mDisplay(X)

##################################################################################################################
##################################################################################################################
##################################################################################################################
#define a function which performs the Gauss-Jordan operation on a matrix

def mSolve(A,B):
    n = len(B)
    for k in range(n):
        if abs(A[k][k]) < 1.0e-6: #to take care of the limitatoins with the float variable storage
            for i in range(k+1, n): 
                if abs(A[i][k]) > abs(A[k][k]):
                    for j in range(k, n): 
                        A[k][j], A[i][j] = A[i][j], A[k][j] 
                    B[k], B[i] = B[i], B[k]
                    break
        pivot = A[k][k]
        if pivot == 0:
            print("Unique solution does not exist.")
            return
        for j in range(k, n):
            A[k][j] /= pivot
        B[k] /= pivot
        for i in range(n):
            if i == k or A[i][k] == 0: 
                continue    #continues to the next iteration and does not print
            factor = A[i][k]
            for j in range(k, n): 
                A[i][j] -= factor * A[k][j]
            B[i] -= factor * B[k]
    if len(A)==4:
        var=['x','y','z','w']
        for i in range(len(var)):
            print(var[i]+' = '+str(B[i])) #prints the result of the Gauss-Jordan Elimination operations
            i += 1
    elif len(A)==3:
        var=['x','y','z']
        for i in range(len(var)):
            print(var[i]+' = '+str(B[i])) #prints the result of the Gauss-Jordan Elimination operations
            i += 1
##################################################################################################################
#define a function which displays the determinant of a matrix

def mDeterminant(A):
    print("the determinant of matrix is ")
    n = len(A)
    if len(A) != len(A[0]): #determinant of only a square matirx is possible, so if no. of row = no. of column, determinant is possible
        print("Input matrix is not square")
    else:
        m = 0 #m will the the number used to calculate the number of swapping. it will give the sign in front of the determinant
        for k in range(n-1):
            if abs(A[k][k]) < 1.0e-12:
                for i in range(k+1, n):
                    if abs(A[i][k]) > abs(A[k][k]):
                        for j in range(k, n):
                            A[k][j], A[i][j] = A[i][j], A[k][j]
                            m = m + 1
            for i in range(k+1, n):
                if A[i][k] == 0: continue
                factor = A[i][k]/A[k][k]
                for j in range(k, n):
                    A[i][j] = A[i][j] - factor * A[k][j]
        t = 1
        for i in range(len(A)):
            t = t * A[i][i]
        print(t*(-1)**m) #printing the result

##################################################################################################################
#define a function which displays the inverse of a matrix using Gauss-Jordan method

def mGJInverse(A):
    print("The inverse of matrix is\n")
    n = len(A)
    X = [[0.0 for i in range(len(A))] for j in range(len(A))] #the augmented matrix is constructed
    for i in range(3):
        for j in range(3):
            X[j][j] = 1.0
    for i in range(len(A)):
        A[i].extend(X[i])   #our input matrix is modified with the augmented matrix
    for k in range(n):
        if abs(A[k][k]) < 1.0e-12:
            for i in range(k+1, n):
                if abs(A[i][k]) > abs(A[k][k]):
                    for j in range(k, 2*n):
                        A[k][j], A[i][j] = A[i][j], A[k][j]
                    break
        p = A[k][k] #pivot p is created
        if p == 0: 
            print("Uninvertible input matrix provided.")
            return
        else:
            for j in range(k, 2*n):
                A[k][j] /= p
            for i in range(n):
                if i == k or A[i][k] == 0: continue
                factor = A[i][k]
                for j in range(k, 2*n):
                    A[i][j] -= factor * A[k][j]
    for i in range(len(A)):
        for j in range(n, len(A[0])):
            print(A[i][j], end = " ") 
        print()

##################################################################################################################
##################################################################################################################
##################################################################################################################
#define a function for forward substitution

def mForwardSubstitution(A,a):
    y = [0 for i in range(len(a))]  #creating an empty list to store values
    for i in range(len(a)):
        j = 0
        for j in range(i):
            j += A[i][j]*y[j]
        y[i] = (a[i] -j)/A[i][i]
    return y

##################################################################################################################
#define a function for backward substitution

def mBackwardSubstitution(B,b):
    x = [0 for i in range(len(b))]   #creating an empty list to store values
    for i in range(len(b)-1, -1, -1):
        j = 0
        for j in range(i+1, len(b)):
            j += B[i][j] * x[j]
        x[i] = (b[i] - j)/B[i][i]
    return x

##################################################################################################################
#define a function for Doolittle's method

def mDoolittle(a):
    n = len(a)
    A = [[0 for i in range(n)] for j in range(n)]   #creating an empty matrix to store values
    B = [[0 for i in range(n)] for j in range(n)]   #creating an empty matrix to store values
    for z in range(n):
        A[z][z] = 1
        a1, a2, a3 = 0, 0 ,0
        for i in range(z):
            a1 += A[z][i]*B[i][z]   
        B[z][z] = (a[z][z] - a1)    #changing the value of the appropriate element in matrix B
        for j in range(z+1, n):
            for i in range(z):
                a2 += A[z][i]*B[i][j]
            B[z][j] = (a[z][j] - a2)     #changing the value of the appropriate element in matrix B
        for k in range(z+1, n):
            for i in range(z):
                a3 += A[k][i]*B[i][z]
            A[k][z] = (a[k][z] - a3)/B[z][z]     #changing the value of the appropriate element in matrix A
    return (A, B)

##################################################################################################################
#define a function for Crout's method

def mCrout(a):
    n = len(a)
    A = [[0 for i in range(n)] for j in range(n)]   #creating an empty matrix to store values
    B = [[0 for i in range(n)] for j in range(n)]   #creating an empty matrix to store values
    for z in range(n):
        B[z][z] = 1
        for j in range(z, n):
            tempL = a[j][z] 
            for k in range(z):
                tempL -= A[j][k]*B[k][z]
            A[j][z] = tempL     #changing the value of the appropriate element in matrix A
        for j in range(z+1, n):
            tempU = a[z][j]
            for k in range(z):
                tempU -= A[z][k]*B[k][j]
            B[z][j] = tempU/A[z][z]     #changing the value of the appropriate element in matrix B
    return (A, B)

##################################################################################################################
#define a function for solving a set of linear equations using Forward or Backward Substitution

def mLinearSolve(A, b, func):       #using either the doolittle or crout method in place of func
    A, B = func(A)
    print("A = " + str(A) + "\n")
    print("B = " + str(B) + "\n")
    y = mForwardSubstitution(A, b)
    x = mBackwardSubstitution(B, y)
    return x

##################################################################################################################
#define a function for Cholesky method

def mCholesky(a):
    n = len(a)
    A = [[0 for i in range(n)] for j in range(n)]      #creating an empty matrix to store values
    for j in range(n):
        for i in range(j, n):
            if i == j:
                sumj = 0
                for k in range(j):
                    sumj = sumj + (A[i][k]**2)
                A[i][j] = (a[i][j] - sumj)**(1/2)       #changing the value of the appropriate element in matrix A
            else:
                sumk = 0
                for k in range(j):
                    sumk = sumk + (A[i][k]*A[j][k])
                A[i][j] = (a[i][j] - sumk)/A[j][j]      #changing the value of the appropriate element in matrix A
    return A

##################################################################################################################
#define a function for solving a set of linear equations using Colesky method

def mCholeskyLinearSolve(A, B, b):
    n = len(A)
    y = [0 for i in range(n)]       #creating an empty list to store values
    x = [0 for i in range(n)]       #creating an empty list to store values
    for i in range(n):
        sumj = 0
        for j in range(i):
            sumj += A[i][j]*y[j]
        y[i] = (b[i]-sumj)/A[i][i]
    for i in range(n-1, -1, -1):
        sumj = 0
        for j in range(i+1, n):
            sumj += B[i][j]*x[j]
        x[i] = (y[i]-sumj)/B[i][i]
    return x

##################################################################################################################
#define a function for finding the Forward Backward Substitution

def mForwardBackwardSubstitution(L, U, b):
    y = [[0 for c in range(len(b[0]))] for r in range(len(b))]      #creating an empty matrix to store values
    for i in range(len(b)):
        for k in range (len(b[0])): 
            y[i][k] = b[i][k]
            for j in range(i):
                y[i][k]=y[i][k]-(L[i][j]*y[j][k]) 
            y[i][k] = y[i][k]/L[i][i]       #changing the appropriate value in our empty matrix
    n = len(y)
    x = [[0,0,0,0] for r in range(len(b))]
    if U[n-1][n-1] == 0: 
        raise ValueError
    for i in range(n-1, -1, -1):
        for k in range (len(b[0])): 
            x[i][k] = y[i][k]
            for j in range(i+1,n):
                x[i][k] = x[i][k] -(U[i][j]*x[j][k]) #changing the appropriate value in our other empty matrix
            x[i][k] = x[i][k]/U[i][i]    
    print ("The inverse of the given matrix is " )
    for i in x:
        print (i)
    return(x)

##################################################################################################################
#define a function for finding the Partial Pivot of a matrix

def mPartialPivot(Ab, m, rows, cols):
    global n,swap          
    n = 0
    swap = 0
    pivot = Ab[int(m)][int(m)]         
    for i in range (int(rows)):         
        if pivot < Ab[int(i)][int(m)]:  
            pivot = Ab[int(i)][int(m)]
            n += 1
            swap = i
    if swap != 0:
        mSwapRows(Ab, m, swap, cols)    
            
    if int(pivot) == 0:
        print ("No unique solution")
        return None

##################################################################################################################
#define a function for finding the inverse of a matrix using LU method

def mInverseLU(M,I): 
    if M[1][1] == 0 and M[0][1] != 0:
        mSwapRows(M, 0,1,4) 
    mLUDecomposition(M)

    L = mLUDecomposition.L 
    U = mLUDecomposition.U 
    return mForwardBackwardSubstitution(L,U,I)

##################################################################################################################
#define a function for swapping the rows of a matrix

def mSwapRows(Ab, old, new_r, cols):
    temp = []       

    for c in range (0, int(cols)):
        temp.append(Ab[int(old)][c])
        Ab[int(old)][c] = Ab[int(new_r)][c]     
        Ab[int(new_r)][c] = temp[c]

##################################################################################################################
#define a function for finding the LU Decomposition of a matrix

def mLUDecomposition(A):
    mPartialPivot(A, 0, len(A), len(A[0]))  
    n = len(A)        
    lower = [[0 for x in range(n)]
             for y in range(n)]
    upper = [[0 for x in range(n)]
             for y in range(n)]
    for i in range(n):
        for j in range(i, n):
            sum = 0
            for k in range(i):
                sum += (lower[i][k] * upper[k][j])
            upper[i][j] =  A[i][j] - sum
        for k in range(i, n):
            if (i == k):
                lower[i][i] = 1  
            else:
                sum = 0
                for j in range(i):
                    sum += (lower[k][j] * upper[j][i])
                lower[k][i] = ((A[k][i] - sum) /
                                  upper[i][i])
    mLUDecomposition.L = lower
    mLUDecomposition.U = upper

##################################################################################################################
##################################################################################################################
##################################################################################################################
import matplotlib.pyplot as plt
import math

##################################################################################################################
#define a function to calculate roots by bisection method
#'func' is the given function, 'a' and 'b' are the lower and upper bounds of the interval and 'e' is the accuracy parameter
def bisection(func, a, b, e):
    i = 0 
    x = 1 
    fxi = [] #list of all f(xi)
    iteration = [] #list of i
    xi = [] #list of xi
    condition = True
    #if function is increasing
    if func(a)<func(b):
        while condition:
            g = func(x)
            x = (a+b)/2
            if func(x)<0:
                a = x
            else:
                b = x
            fxi.append(func(x))
            iteration.append(x)
            i = i + 1
            xi.append(i)
            condition = abs(func(x)-g)>e #the condition is checked and truthval is set
    #if function is decreasing
    if func(a)>func(b):
        while condition:
            g = func(x)
            x = (a+b)/2
            if func(x)<0:
                b = x
            else:
                a = x
            fxi.append(func(x))
            iteration.append(x)
            i = i + 1
            xi.append(i)
            condition = abs(func(x)-g)>e #the condition is checked and truthval is set       
    print("The root is ", x) 
    return xi, fxi, iteration

##################################################################################################################
#define a function to display root, plot graph and table
def rBisection(y, a, b):
    if y(a)*y(b)<0: 
        xi, fxi, iteration = bisection(y, a, b, 1.0e-5) 
        x1 = xi
        y1 = fxi
        y11 = iteration
        plt.plot(x1, y1)
        plt.ylabel("f(xi)")
        plt.xlabel("i")
        plt.show()
        print("i", end=" ")
        print("xi")
        for i in range(len(x1)):
            print(x1[i], end=" ")
            print(y11[i])
    else: #we take beta=1.2
        if abs(y(a))<abs(y(b)):
            a = a - 1.2*(b-a)
            rBisection(y, a, b) 
        if abs(y(a))>abs(y(b)):
            b = b + 1.2*(b-a)
            rBisection(y, a, b) 

##################################################################################################################
#define a function to calculate the root by regula falsi method
#'func' is the given function, 'a' and 'b' are the lower and upper bounds of the interval, 'e' is the accuracy parameter and 'n' is the maximum number of iterations
def regulafalsi(f,a,b,e,n):
    x = 0 
    fxi = [] 
    iteration = []
    for fal in range(1,n+1):
        x = b - (b-a)/(f(b)-f(a))*f(b)
        fxi.append(f(x))
        iteration.append(x)
        if abs(f(x)) < e: break
        elif f(a)*f(x)<0:
            b = x
        else:
            a = x
    print("Required root is: ", x)
    return fxi, iteration

##################################################################################################################
#define a function to display root, plot graph and table
def rRegulaFalsi(y, a, b):
    if y(a)*y(b)<0:
        iterCount, iteration = regulafalsi(y, a, b, 1.0e-5, 100)
        x2 = list(range(1, len(iterCount)+1))
        y2 = iterCount
        y22 = iteration
        plt.plot(x2, y2)
        plt.ylabel("f(xi)")
        plt.xlabel("i")
        plt.show()
        print("i", end=" ")
        print("xi")
        for i in range(len(x2)):
            print(x2[i], end=" ")
            print(y22[i])
    else:
        if abs(y(a))<abs(y(b)): #beta=1.2
            a = a - 1.2*(b-a)
            rRegulaFalsi(y, a, b)
        if abs(y(a))>abs(y(b)):
            b = b + 1.2*(b-a)
            rRegulaFalsi(y, a, b)

##################################################################################################################
#define a function to calculate roots by Newton-Raphson method
def rNewtonRaphson(f, df, D, e, n): #'f' is the given function, 'df' is its derivative, 'D' is the dummy variable,'e' is the accuracy parameter and 'n' is the maximum number of iterations
    iterCount = [f(D)]
    iteration = []
    for i in range(n):
        xx = D - f(D)/df(D)
        iterCount.append(f(xx))
        iteration.append(xx)
        if abs(xx - D)<e: break
        D = xx
    x3 = list(range(1, len(iterCount)+1))
    y3 = iterCount
    y33 = iteration
    plt.plot(x3, y3)
    plt.show()
    print("i", end=" ")
    print("x_i")
    for i in range(len(x3)-1):
        print(x3[i], end=" ")
        print(y33[i])
    return xx, i

##################################################################################################################
#define a function to find the co-efficients of the differentials
#p = a[0] + a[1]*x + a[2]*xˆ2 +...+ a[n]*xˆn
def polynomialcoefficients(a, x):
    n = len(a) - 1
    p = a[n]
    dp = 0.0
    d2p = 0.0
    for i in range(1, n+1): #dp = first derivative of p and d2p = second derivative of p
        d2p = d2p*x +2.0*dp
        dp = dp*x + p
        p = p*x + a[n-i]
    return p, dp, d2p

##################################################################################################################
#define a function to calculate roots by Laguerre's method
#roots = polyRoots(a).
def rLaguerre(U, a, e=1.0e-12): #'U' is the guess root, 'a' is the list of coefficients and 'e' is the accuracy parameter
    def laguerre(a, e):
        x = U
        n = len(a) - 1
        for i in range(30):
            p, dp, d2p = polynomialcoefficients(a,x)
            if abs(p) < e: return x
            g = dp/p
            h = g*g - d2p/p
            f = math.sqrt((n-1)*(n*h - g*g))
            if abs(g+f) > abs(g-f): dx = n/(g+f)
            else: dx = n/(g-f)
            x = x - dx
            if abs(dx) < e: return x
        print("Input another Guess") #just in case the guess is too off
    def deflatepolynomial(a, root):
        n = len(a) - 1
        b = [0.0]*n
        b[n-1] = a[n]
        for i in range(n-2, -1, -1):
            b[i] = a[i+1] + root*b[i+1]
        return b
    n = len(a) - 1
    roots = [0.0 for i in range(n)]
    for i in range(n):
        x = laguerre(a, e)
        roots[i] = x
        a = deflatepolynomial(a, x)
    return roots

##################################################################################################################
##################################################################################################################
##################################################################################################################

import random
##################################################################################################################
#define a function for numerical integration by Midpoint method
def iMidpoint(a, b, N, fx): #a, b are the limits of integration, N is the no. of divisions and fx is the integrand   
    h = (b-a)/N
    midpoints = []
    for i in range(1, N+1):
        midpoints.append((2*a + (2*i -1)*h)/2)
    integral = 0
    for j in range(len(midpoints)):
        integral = integral + h*fx(midpoints[j])
    return integral

##################################################################################################################
#define a function for numerical integration by Trapezoidal method
def iTrapezoidal(a, b, N, fx): #a, b are the limits of integration, N is the no. of divisions and fx is the integrand
    h = (b-a)/N
    endpoints = []
    for i in range(N+1):
        endpoints.append(a + i*h)
    tr = []
    for j in range(1, len(endpoints)):
        tr.append((h/2)*(fx(endpoints[j-1])+fx(endpoints[j])))
    integral = sum(tr)
    return integral

##################################################################################################################
#define a function for numerical integration by Simpson method
def iSimpson(a, b, N, fx):#a, b are the limits of integration, N is the no. of divisions and fx is the integrand
    if (N % 2) != 0:    #convert N to even if N is odd
        N = N + 1
    h = (b-a)/N
    endpoints = []
    for i in range(N+1):
        endpoints.append(a + i*h)
    si = []
    for j in range(0, N-1, 2):
        si.append((h/3)*(fx(endpoints[j])+(4*fx(endpoints[j+1]))+fx(endpoints[j+2])))
    integral = sum(si)
    return integral

##################################################################################################################
#define a function for numerical integration by Monte Carlo method
def iMonteCarlo(a, b, N, d, iterations, fx): #a, b are the limits of integration, N is the no. of divisions and fx is the integrand
 #d is the value by which N is increaased and iterations is the number of times N is be increased            
    yaxis = []
    xaxis = list(range(N, iterations*d, d))
    while N < (iterations*d):
        vars = []
        for i in range(N):
            vars.append(random.uniform(a, b))
        f = []
        f2 = []
        for j in range(len(vars)):
            f.append(fx(vars[j]))
        for j in range(len(vars)):
            f2.append((fx(vars[j]))**2)
        sumf = sum(f)
        sumf2 = sum(f2)
        F = ((b-a)/N)*sumf
        yaxis.append(F)
        SDf = math.sqrt((1/N)*sumf2 - ((1/N)*sumf)**2)
        N = N + d
    plt.plot(xaxis, yaxis)
    plt.ylabel("Value of pi")
    plt.xlabel("N")
    plt.show

##################################################################################################################
##################################################################################################################
##################################################################################################################
