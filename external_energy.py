import cv2
from skimage.filters import gaussian
import numpy as np

def line_energy(image):
    #implement line energy (i.e. image intensity)
    return image

    #raise NotImplementedError
def edge_energy(image):
    sx = np.array([[-1, 0, 1]])
    sy = np.array([[-1, 0, 1]]).T

    X = cv2.filter2D(image, cv2.CV_64F, sx)
    Y = cv2.filter2D(image, cv2.CV_64F, sy)
    #print(f" Before X is {X} and Y is {Y}")
    X, Y = np.gradient(image)
    #print(f" After X is {X} and Y is {Y}")


    E = ((X**2)+(Y**2))**0.5
    E = (E-np.amin(E))/(np.amax(E) - np.amin(E))

    print(f"edge energy {E}")
    return E

    #implement edge energy (i.e. gradient magnitude)
    #raise NotImplementedError
def term_energy(image):
    sx = np.array([[-1, 0, 1]])
    sy = np.array([[-1, 0, 1]]).T

    
    X = cv2.filter2D(image, cv2.CV_64F, sx)
    Y = cv2.filter2D(image, cv2.CV_64F, sy)
    X, Y = np.gradient(image)

    XX = cv2.Sobel(image, cv2.CV_64F , dx = 2, dy= 0)
    YY = cv2.Sobel(image, cv2.CV_64F , dx = 0, dy= 2)
    XY = cv2.Sobel(X, cv2.CV_64F , dx = 0, dy= 1)


    E =cv2.divide((XX*(Y**2)+YY*(X**2) - (2*XY*X*Y)), (((X**2 + Y**2)**1.5) +1e-8))
    #print(E)
    E[np.isnan(E)] = 0.0

    E = (E - np.amin(E))/(np.amax(E)-np.amin(E))
    print(f" Normalised Term Energy {E}")

    return E
    #implement term energy (i.e. curvature)
    #raise NotImplementedError
def external_energy(image, w_line, w_edge, w_term):
    #implement external energy
    E = w_line*line_energy(image) + w_edge*edge_energy(image) + w_term*term_energy(image)
    E = (E - np.amin(E))/(np.amax(E)-np.amin(E))
    # cv2.imshow(f'External Energy Normalized ({w_line}, {w_edge}, {w_term})', E)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(E)
    return E

    