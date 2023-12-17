import numpy as np  
import cv2

def get_matrix(alpha, beta, gamma, num_points):
    """Return the matrix for the internal energy minimization.
    # Arguments
        alpha: The alpha parameter.
        beta: The beta parameter.
        gamma: The gamma parameter.
        num_points: The number of points in the curve.
    # Returns
        The matrix for the internal energy minimization. (i.e. A + gamma * I)
    """

    baseRow = np.zeros(num_points)
    baseRow[0:3] = ((2*alpha + 6*beta) , (-alpha - 4*beta) ,  beta)
    baseRow[num_points-2 :num_points] =(beta,(-alpha - 4*beta)) 


    A = np.zeros((num_points, num_points))
    for i in range(num_points):
        A[i] = np.roll(baseRow , i)
    I = np.eye(num_points)
    M = np.linalg.inv(A+(gamma*I))
    
    # cv2.imshow(f'Internal Energy Matrix ({alpha}, {beta}, {gamma})', M)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    print(M)
    return M





    raise NotImplementedError
