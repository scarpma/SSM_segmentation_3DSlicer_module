import numpy as np

import nibabel as nib
import pyvista as pv
import vtk
import scipy.spatial.distance
import scipy.interpolate
import matplotlib.pyplot as plt

import os
import os.path as osp

inputVolumePathTemplate = "/home/bcl/Martino/aorta_segmentation_v3/aorta_dataset_nii/imagesTr/{}.nii.gz"
convertSSMModelsToRAS = True
SSMModelPathTemplate = "/home/bcl/Martino/aortaRegistrationPytorch3d/aligned_dataset_meshmixer_corrected_relaxed_smoothedBou/V2_to_{}_relaxed.vtp"
origModelPathTemplate = "/home/bcl/Martino/aorta_segmentation_v3/aorta_dataset_nii/simulationModelsTr/{}.vtp"
profileSpacing = 1. #mm
profileLen = 20 #mm
profilesDir = "./profiles/"
plots = True

def ICP(source, target):
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(source.copy())
    icp.SetTarget(target.copy())
    icp.GetLandmarkTransform().SetModeToRigidBody()
    #icp.DebugOn()
    icp.SetMaximumNumberOfIterations(20)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(source)
    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()

    return pv.PolyData(icpTransformFilter.GetOutput())

def getGreedyPerm(D, M):
    """
    A Naive O(N^2) algorithm to do furthest points sampling
    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points
    Return
    ------
    tuple (list, list)
        (permutation (N-length array of indices),
        lambdas (N-length array of insertion radii))
    preso da
    https://gist.github.com/ctralie/128cc07da67f1d2e10ea470ee2d23fe8
    """
    N = D.shape[0]
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(M, dtype=np.int64)
    lambdas = np.zeros(M)
    ds = D[0, :]
    for i in range(1, M):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)

def extractProfiles(patientName):

    def convertRasToIjk(points):
        points = np.c_[points, np.ones(points.shape[0])]
        pointsIjk = np.einsum('ab,ib->ia', RASToIjk, points)[:,:-1]
        return pointsIjk

    inputVolumePath = inputVolumePathTemplate.format(patientName)
    inputVolumeNib = nib.load(inputVolumePath)
    print("Loading nifti...", end=" ")
    inputVolumeNumpy = inputVolumeNib.get_fdata()
    print("done!")
    ijkToRASMatrix = inputVolumeNib.affine
    RASToIjk = np.linalg.inv(ijkToRASMatrix)

    # define interpolation method for the image volume
    ii, jj, kk = [np.arange(0,inputVolumeNumpy.shape[i]) for i in range(3)]
    interpolator = scipy.interpolate.RegularGridInterpolator(
        (ii, jj, kk),
        inputVolumeNumpy,
        method='linear',
        bounds_error=False,
        fill_value=0.,
    )

    SSMModelPath = SSMModelPathTemplate.format(patientName)
    SSMModel = pv.read(SSMModelPath)
    LPSToRASMatrix = np.diag([-1,-1,1,1])
    if convertSSMModelsToRAS:
        SSMModel = SSMModel.transform(LPSToRASMatrix)
    origModelPath = origModelPathTemplate.format(patientName)
    origModel = pv.read(origModelPath)
    if convertSSMModelsToRAS:
        origModel = origModel.transform(LPSToRASMatrix)
    SSMTransformed = ICP(SSMModel, origModel)
    SSMTransformed = SSMTransformed.compute_normals(cell_normals=False)

    #SSMTransformed.save('SSMTrasformed.vtp')

    # # define sample points
    # distances = scipy.spatial.distance.pdist(SSMTransformed.points , metric='euclidean')
    # distances = scipy.spatial.distance.squareform(distances, force='tomatrix', checks=True)
    # nSamplePoints = 500
    # samplePointsIdxs = getGreedyPerm(distances, nSamplePoints)[0]
    # samplePoints = SSMTransformed.points[samplePointsIdxs]
    # np.savetxt('samplePoints.txt', samplePointsIdxs, fmt="%i", delimiter=',')
    # pv.PolyData(samplePoints).save('samplePointsForA2Model.vtp')

    samplePointsIdxs = np.loadtxt('samplePoints.txt', dtype=int)
    samplePoints = SSMTransformed.points[samplePointsIdxs]
    sampleNormals = SSMTransformed['Normals'][samplePointsIdxs]

    def computeProfilePoints(point, direction):
        t = np.arange(-profileLen/2., (profileLen)/2 + profileSpacing, profileSpacing)
        # (t.shape[0], nSamplePoints, 3)
        return point + np.einsum('i,kj->ikj', t, direction)

    # compute coordinates of sampling points (near the model's surface)
    profilePoints = computeProfilePoints(samplePoints, sampleNormals)
    # convert sampling points coordinate to IJK space to sample the image
    pointsIjk = convertRasToIjk(profilePoints.reshape(-1,3))
    profiles = interpolator(pointsIjk).reshape(profilePoints.shape[:-1])

    #profiles_norm_grad = np.gradient(profiles, profileSpacing, axis=0)
    #profiles_norm_grad = np.gradient(profiles, profileSpacing, axis=0)
    profiles_norm_grad = profiles

    if not osp.isdir(profilesDir):
        os.makedirs(profilesDir)
    profilePathTemplate = osp.join(profilesDir, "{}.txt")
    np.savetxt(profilePathTemplate.format(patientName), profiles_norm_grad.T.reshape(-1), fmt='%.5f')

    # plots
    profilesPlotsDir = osp.join(profilesDir, "plots")
    if not osp.isdir(profilesPlotsDir):
        os.makedirs(profilesPlotsDir)
    plotsPathTemplate = osp.join(profilesPlotsDir, "{}.png")
    if plots:
        t = np.arange(-profileLen/2., (profileLen)/2 + profileSpacing, profileSpacing)
        plt.errorbar(t, profiles_norm_grad.mean(-1), yerr=profiles_norm_grad.std(-1))
        plt.xlabel("displacement from surface [mm]")
        plt.ylabel("normalized gradient")
        plt.title("{} profiles mean".format(patientName))
        plt.savefig(plotsPathTemplate.format(patientName), dpi=80)
        plt.close()



if __name__ == "__main__":

    import sys
    assert len(sys.argv) >= 2

    if sys.argv[1] == 'one':
        assert(len(sys.argv)==3)
        extractProfiles(sys.argv[2])
    elif sys.argv[1] == 'all':
        import glob
        import parse

        def find_matches(format_str):
            expansions = glob.glob(format_str.format("*"))
            matches  = [parse.parse(format_str, exp)[0] for exp in expansions]
            return sorted(matches)

        def compare_found_patients(*lists):
            sets = [set(l) for l in lists]
            union = set()
            union = union.union(*sets)
            #print(union)
            intersection = union.copy()
            intersection = intersection.intersection(*sets)
            print(f"{len(intersection)} valid patients")
            for ii, s in enumerate(sets):
                print(f"unique patients of set {ii}:")
                print(s-intersection)
            return sorted(list(intersection))

        def printPatientNames(l):
            for i in range(len(l)):
                if i%5!=0 or i==0 :
                    print("{:4s}".format(l[i]), end=" ")
                else:
                    print("{:4s}".format(l[i]))

        volumePatients = find_matches(inputVolumePathTemplate)
        modelPatients = find_matches(origModelPathTemplate)
        SSMPatients = find_matches(SSMModelPathTemplate)
        patientNames = compare_found_patients(volumePatients, modelPatients, SSMPatients)
        printPatientNames(patientNames)

        for patientName in patientNames:
            print(patientName)
            extractProfiles(patientName)
    else:
        print("sys.argv[1] must be 'one' or 'all'")













