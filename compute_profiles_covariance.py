import numpy as np
import glob
import sklearn.covariance as Covariance

def get_covariance_object(X, load=True):
    if load:
        covarianceDict = np.load('./profiles/covarianceDict.npy', allow_pickle=True)[()]
        cov_object, mean = covarianceDict['cov_object', 'mean']
        return cov_object, mean

    mean = X.mean(0)
    cov_object = Covariance.OAS(assume_centered=True).fit(X-mean)
    #cov_object = Covariance.EmpiricalCovariance(assume_centered=True).fit(X-mean)
    #cov_object = Covariance.ShrunkCovariance(assume_centered=True, shrinkage=0.01).fit(X-mean)
    #cov_object = MinCovDet(assume_centered=True).fit(X-mean)
    #cov_object = Covariance.GraphicalLassoCV(assume_centered=True).fit(X-mean)
    covarianceDict = {
        "cov_object" : cov_object,
        "mean" : mean,
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
    return cov_object, mean

def mahalanobis(cov_object, X, mean):
    return cov_object.mahalanobis(X-mean)


if __name__ == "__main__":

    import sys
    assert len(sys.argv) == 2
    assert sys.argv[1] in ['load', 'save']

    profilesPaths = glob.glob("./profiles/*.txt")

    profiles = []
    for p in profilesPaths:
        profiles.append(np.loadtxt(p))

    profiles = np.stack(profiles)[:,:]

    #rank of covariance matrix
    #(scipy.linalg.svdvals(np.dot(p.T, p)) > 1e-8).sum()
    load = True if sys.argv[1] == 'load' else False
    cov_object, mean = get_covariance_object(profiles, load=load)

