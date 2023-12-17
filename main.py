import cv2
import numpy as np
from scipy import interpolate     
import matplotlib.pyplot as plt
from external_energy import external_energy
from internal_energy_matrix import get_matrix
from skimage.filters import gaussian
import math


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        #save point
        xs.append(x)
        ys.append(y)

        #display point
        cv2.circle(img, (x, y), 3, 128, -1)
        cv2.imshow('image', img)





if __name__ == '__main__':
    #point initialization
    img_path = '../images/star.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f" image {img}")

    xs = []
    ys = []
    cv2.imshow('image', img)
    image = img.copy()
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #selected points are in xs and ys
    print(xs)
    print(ys)
    print("xs and ys above ----------------------------")

    #implement part 1: interpolate between the  selected points
    n=2000
    
    #print(img.shape)
    #print(img.shape[0])
    #print(img.shape[1])
    #img[0][1]=1
    #print(img)
    #print(img[0][1])
    xs.append(xs[0])
    ys.append(ys[0])
    xs  = np.array(xs)
    ys  = np.array(ys)
    print(xs)
    print(ys)
    print("xs and ys above ----------------------------")
    print(xs.shape)
    contours  = np.zeros((xs.shape[0] , 2))

    contours[: , 0] = xs[:]
    contours[: , 1] = ys[:]
    tck, u = interpolate.splprep(contours.T, u=None, k=1, per=1)
    u_new = np.linspace(u.min(), u.max(), n)
    xs_new, ys_new = interpolate.splev(u_new, tck, der=0)

    
    contours = np.zeros((len(xs_new), 2))
    contours[:, 0] = xs_new[:]
    contours[:, 1] = ys_new[:]
    contours = contours.reshape((-1, 1, 2)).astype(np.int32)
    
    smoothed = cv2.GaussianBlur(image, (5,5),3)
    
    #Basic ones to avoid error if all other are commented
    alpha = 2
    beta = 1.
    gamma = 1.0
    kappa = 1.5

    w_line = 0.5
    w_edge = 0.5
    w_term = 1.5

    iterations = 6000

    #Square Finalised
    # alpha = 10
    # beta = 1.0
    # gamma = 1.0
    # kappa = 3.0

    # w_line = -0.5
    # w_edge = 2.0
    # w_term = 5

    
    #Star  Finalised
    alpha = 0.00
    beta = 0.0
    gamma = 1.0
    kappa = 1.0

    w_line = -0.5
    w_edge = 2.0
    w_term = 1.5

    smoothed = cv2.bitwise_not(smoothed)   
    '''Added to invert image, code works better with start being black'''
    
    
    #shape 
    # alpha = 0.1 
    # beta = 0.7
    # gamma = 1.0
    # kappa = 1.5

    # w_line = -0.5
    # w_edge = 0.5
    # w_term = 3



    #Circle Finalised
    # alpha = 5
    # beta = 1.0
    # gamma = 1.0
    # kappa = 0.1 

    # w_line = -0.5
    # w_edge = 0.5
    # w_term = 1.5


    #Vase mostly final
    # alpha = 10
    # beta = 0.5
    # gamma = 1.0
    # kappa = 0.4

    # w_line = 0.5
    # w_edge = 2.0
    # w_term = 5.0
    
    # #To show contour on vase image
    # vaseimg = img.copy()
    # cv2.drawContours(vaseimg, contours, -1, (255, 95, 255), 3, cv2.LINE_AA)
    # cv2.imshow("vase contour",vaseimg)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    


    #Dental 
    # alpha = 2.0
    # beta = 1.0
    # gamma = 1.0
    # kappa = 1.5

    # w_line = -0.5
    # w_edge = 1.0
    # w_term = 1.5


    #brain outter
    # alpha = 5
    # beta = 1.0
    # gamma = 1.0
    # kappa = 1.5

    # w_line = 0.5
    # w_edge = 0.5
    # w_term = 1.5
    

    #brain Inner
    # alpha = 0.5
    # beta = 0.5
    # gamma = 0.5
    # kappa = 1.5

    # w_line = -0.5
    # w_edge = 1.0
    # w_term = 1.5

    #Eye Hole
    # alpha = 5
    # beta = 0.5
    # gamma = 1.0
    # kappa = 0.1

    # w_line = -0.5
    # w_edge = 0.5
    # w_term = 1.5


    num_points = len(xs_new)
    
    #get matrix
    M = get_matrix(alpha, beta, gamma, num_points)

    #get external energy
    
    E = external_energy(smoothed, w_line, w_edge, w_term)

    

    fx = cv2.Sobel(E, cv2.CV_64F, 1, 0, ksize=1)
    fy = cv2.Sobel(E, cv2.CV_64F, 0, 1, ksize=1)
    bfx = np.zeros(n)
    bfy = np.zeros(n)

   
    #fx = interpolate.RegularGridInterpolator(list(zip([xs_new,ys_new])),fx)
       
    #fy = interpolate.RegularGridInterpolator(list(zip([xs_new,ys_new])), fy)
    


    print(f" fx is : {fx}")
    print(f"fy is :{fy}")
    #bfx = interpolate.interp2d(xs_new, ys_new ,  fx)
    #bfy = interpolate.interp2d(xs_new, ys_new ,  fy)
    for i in range(iterations):
        if i%10==0:
            print(f'iteration number: {i}')

        # Boundary Checks
        #print(xs_new.shape)
        #print(ys_new.shape)
        xs_new = np.clip(xs_new , 0 ,img.shape[0]-1)
        ys_new = np.clip(ys_new , 0 ,img.shape[1]-1)
        #print("clipped")
        #print(xs_new.shape)
        #print(ys_new.shape)
        # Define gradients for external energy in x and y direction
    
        
        
        
       
        
        
        x_new = np.dot(M, ((gamma * xs_new) - (kappa * fx[ys_new.astype(np.int32) , xs_new.astype(np.int32)])))
        y_new = np.dot(M, ((gamma * ys_new) - (kappa * fy[ys_new.astype(np.int32) , xs_new.astype(np.int32)])))
        xs_new = x_new.copy()
        ys_new = y_new.copy()

      
        # plt.imshow(image)
        # plt.plot(xs_new,ys_new)
        # plt.pause(0.001)
        # plt.cla()

    img2 = image.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    contours = np.zeros((len(xs_new), 2))
    contours[:, 0] = xs_new[:]
    contours[:, 1] = ys_new[:]
    contours = contours.reshape((-1, 1, 2)).astype(np.int32)
    cv2.drawContours(img2, contours, -1, (255, 95, 255), 3, cv2.LINE_AA)

    cv2.imshow(f'Iterative Contours Snake: wline {w_line}  , wedge {w_edge} , wterm {w_term}', img2)

    cv2.waitKey(0) 
    cv2.destroyAllWindows()