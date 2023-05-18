import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([3, 6, 9, 12, 15])

w = 0

lr = 0.01

def forward(X,Y,w):
    
    y_predicted = np.dot(X,w)
    

    
    return y_predicted
def predict(x):
    
    y_predicted = np.dot(x,w)
    

    
    return y_predicted
    
def loss(Y,y_predicted):
    
    loss = ((Y-y_predicted)**2).mean()
    
    return loss
def gradient(X,Y,_y_predicted):
    
    dw = np.dot(2*X,y_predicted-Y).mean()
    
    return dw
    


if __name__=="__main__":

    for epoch in range(10):
        
        y_predicted = forward(X,Y,w)
        
        mse = loss(Y,y_predicted)
        
        dw = gradient(X,Y,y_predicted)
        
        w -= dw*lr 
        
        print(f"mse {mse}")
        
        print(predict(5))
    