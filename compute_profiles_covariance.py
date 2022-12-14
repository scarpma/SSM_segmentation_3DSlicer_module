import numpy as np
import glob
import sklearn.covariance as Covariance

def get_covariance_object(X, load=True):
    if load:
        covarianceDict = np.load('./profiles/covarianceDict.npy', allow_pickle=True)[()]
        cov_object, mean, std = covarianceDict['cov_object'], covarianceDict['mean'], covarianceDict['std']
        return cov_object, mean, std

    mean = X.mean(0)
    std = X.std()
    X = (X - mean) / std
    cov_object = Covariance.OAS(assume_centered=True).fit(X)
    #cov_object = Covariance.EmpiricalCovariance(assume_centered=True).fit(X)
    #cov_object = Covariance.ShrunkCovariance(assume_centered=True, shrinkage=0.01).fit(X)
    #cov_object = MinCovDet(assume_centered=True).fit(X)
    #cov_object = Covariance.GraphicalLassoCV(assume_centered=True).fit(X)
    covarianceDict = {
        "cov_object" : cov_object,
        "mean" : mean,
        "std" : std,
    }
    np.save('./profiles/covarianceDict.npy', covarianceDict)

    #i = 300
    #G=10
    #plt.title("covariance matrix")
    #plt.imshow(oas.covariance_[21*i:21*(i+G),21*i:21*(i+G)])
    #plt.colorbar()
    #plt.show()
    #
    #i = 400
    #G=5
    #plt.plot(profiles[0][21*i:21*(i+G)], label="sample")
    #plt.plot(profiles.mean(0)[21*i:21*(i+G)], label="average")
    #plt.legend()
    #plt.show()
    return cov_object, mean, std

def mahalanobis(cov_object, X, mean, std):
    return cov_object.mahalanobis( (X-mean) / std )

def loadProfiles():
    profilesPaths = glob.glob("./profiles/*.txt")
    profilesPaths = [p for p in profilesPaths if 'samplePoints' not in p]
    profiles = []
    for p in profilesPaths:
        profiles.append(np.loadtxt(p))
    return np.stack(profiles)[:,:]


if __name__ == "__main__":

    import sys
    assert len(sys.argv) == 2
    assert sys.argv[1] in ['load', 'save']

    profiles = loadProfiles()
    #rank of covariance matrix
    #(scipy.linalg.svdvals(np.dot(p.T, p)) > 1e-8).sum()
    load = True if sys.argv[1] == 'load' else False
    cov_object, mean, std = get_covariance_object(profiles, load=load)

