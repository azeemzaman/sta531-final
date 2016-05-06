F = np.array([[1,0,0],[dt, 1, 0],[0,dt,1]])
H = np.array([[1,0,0],[0,0,1]])
Q = np.array([[100*dt,0,0],[0,dt,0],[0,0,dt]])
R = np.array([[sigma_a**2,0],[0,sigma_p**2]])

def filter_smoother(xs, mu_0, V_0, F, Q, H, R):
    """
    The function implements the Kalman filter and smoother.
    Args:
        xs:  the data
        mu_0: initial values
        V_0: V_0 matrix
        F: F matrix
        Q:  Q matrix
        H: H matrix
        R: R matrix
    
    returns:
        mus:  the means of p(z_j|x_1:j)
        Vs:  the covaraince matrices of p(z_j|x_1:j)
        mu_hats: the mean of p(z_j|x_1:n)
        V_hats: the covariance matrices of p(z_j|x_1:n)
    """
    # get sizes
    N = len(xs)
    size = H.shape[1]
    # create empty vectors to hold results
    Ks = [None] * N
    mus = [None] * N
    Vs = [None] * N
    Ps = [None] * N
    # compute initial values
    Ks[0] = V_0.dot(H.T.dot(np.linalg.inv(H.dot(V_0.dot(H.T)) + R)))
    mus[0] = mu_0 + Ks[0].dot(xs[0] - H.dot(mu_0))
    Vs[0] = (np.eye(size) - Ks[0].dot(H)).dot(V_0)
    Ps[0] = F.dot(Vs[0].dot(F.T)) + Q
    # use recursions to calculate rest of values
    for j in range(1,N):
        Ks[j] = Ps[j-1].dot(H.T.dot(np.linalg.inv(H.dot(Ps[j-1].dot(H.T)) + R)))
        mus[j] = F.dot(mus[j-1]) + Ks[j].dot(xs[j] - H.dot(F.dot(mus[j-1])))
        Vs[j] = (np.eye(size) - Ks[j].dot(H)).dot(Ps[j-1])
        Ps[j] = F.dot(Vs[j].dot(F.T)) + Q
    
    # smoother
    # create empty vectors to hold results
    C = [None] * N
    mu_hats = [None] * N
    V_hats = [None] * N
    # collect initial values
    mu_hats[N-1] = mus[N-1]
    V_hats[N-1] = Vs[N-1]
    # use recursions to calculate other values
    for j in range(N-2, -1, -1):
        C[j] = Vs[j].dot(F.T.dot(np.linalg.inv(Ps[j])))
        mu_hats[j] = mus[j] + C[j].dot(mu_hats[j+1] - F.dot(mus[j]))
        V_hats[j] = Vs[j] + C[j].dot((V_hats[j+1]-Ps[j]).dot(C[j].T))
    return mus, Vs, mu_hats, V_hats
