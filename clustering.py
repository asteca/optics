
import numpy as np
from scipy.spatial import distance
from sklearn.decomposition import PCA
import sklearn.cluster as skclust
from pathlib import Path
from astropy.io import ascii
from astropy.table import Table


"""
This method uses the OPTICS algorithm to estimate membership probabilities
when the cluster occupies most of the observed frame, and thus there is no
identifiable overdensity in the (x, y) coordinates space.

Both pyUPMASK and the Bayesian method in ASteCA require that the cluster
should show an overdensity in the coordinates space, and hence are not
appropriate for the cases handled by this method.

The OPTICS[1] algorithm in this method works on the parallax and proper motions
space, looking for stars in regions of overdensity. Hence, no field region
in coordinates space is required at all.


[1]:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
"""

# Fixed parameters
# IDs column
id_col = "ID"
# These columns will be used by the membership probabilities method
data_cols = ("pmRA", "pmDE", "Plx")
err_cols = ("e_pmRA", "e_pmDE", "e_Plx")
# Number of PCA dimensions to use
PCAdims = 3

# Define the 'min_samples' values used by OPTICS as: (min, max, step)
min_samples_rng = (10, 60, 2)
# Number of times the data will be re-sampled (given its uncertainties) and
# processed again with OPTICS
Nruns = 5

# This is the most important parameter in the method, as it determines how the
# final 'eps' value is selected. The 'perc_cut' parameter means
# "percentile cut", and is used to determine when the members selected by a
# given 'eps' value are beginning to be too "spread out". When this happens,
# then the final ("optimal") eps value has been found.
# 90% is a heuristic value that proved to give reasonable results for various
# clusters.
perc_cut = 90


def main():
    """
    """
    # Process all files in 'input_folder'
    files = readFiles()
    for file_path in files:
        # Extract required data from file
        data_all, data_id, data_c, data_err, msk_accpt = dataExtract(file_path)

        # This dictionary will hold all the runs
        probs_dict = {"ID": data_id}

        # For all the 'min_samples' values in 'min_samples_rng'
        for min_samples in np.arange(
                min_samples_rng[0], min_samples_rng[1], min_samples_rng[2]):
            print("min_sample={}".format(min_samples))

            # For all the re-sample runs
            probs_all = []
            for _ in range(Nruns):
                print("  Re-sample N={}".format(_))

                # Re-sample data
                data_arr = reSampleData(data_c, data_err)
                # Apply PCA reduction
                data_pca = dimReduc(data_arr, PCAdims)

                #
                model_OPTIC = runOPTICS(data_pca, min_samples)

                # Auto eps selection
                eps_final = findEps(data_pca, model_OPTIC, perc_cut)

                # DBSCAN labels
                labels_dbs = skclust.cluster_optics_dbscan(
                    reachability=model_OPTIC.reachability_,
                    core_distances=model_OPTIC.core_distances_,
                    ordering=model_OPTIC.ordering_, eps=eps_final)

                msk_memb = labels_dbs != -1
                probs = np.zeros(len(msk_accpt))
                j = 0
                for i, st_f in enumerate(msk_accpt):
                    if st_f:
                        if msk_memb[j]:
                            probs[i] = 1
                        j += 1
                probs_all.append(probs)

            probs_dict[str(min_samples)] = np.round(np.mean(probs_all, 0), 3)

        # Estimate mean probabilities
        all_vals = []
        for k, vals in probs_dict.items():
            if k != 'ID':
                all_vals.append(vals)
        probs_mean = np.round(np.array(all_vals).mean(0), 3)

        # Write to file
        probs_dict['probs_mean'] = probs_mean
        fname, fext = file_path.parts[-1].split('.')
        fout = 'output/' + fname + "_probs." + fext
        ascii.write(probs_dict, fout, overwrite=True)


def findEps(data, model_OPTIC, perc_cut, eps_step=0.005):
    """
    DBSCAN Python Example: The Optimal Value For Epsilon (EPS)
    https://towardsdatascience.com/
    machine-learning-clustering-dbscan-determine-the-optimal-value-for-
    epsilon-eps-python-example-3100091cfbc

    Amin Karami and Ronnie Johansson. Choosing dbscan parameters
    automatically using differential evolution. International Journal
    of Computer Applications, 91(7), 2014

    Clustering Using OPTICS
    https://towardsdatascience.com/clustering-using-optics-cac1d10ed7a7

    OPTICS attributes used here:

    labels_: array, shape (n_samples,)
     Cluster labels for each point in the dataset given to fit(). Noisy
     samples and points which are not included in a leaf cluster of
     cluster_hierarchy_ are labeled as -1.

    reachability_: array, shape (n_samples,)
     Reachability distances per sample, indexed by object order. Use
     model.reachability_[model.ordering_] to access in cluster order.

    ordering_: array, shape (n_samples,)
     The cluster ordered list of sample indices.

    """

    def mskNoise(eps_v):
        # Extract the labeled assigned to each point for this eps value
        labels_dbs = skclust.cluster_optics_dbscan(
            reachability=model_OPTIC.reachability_,
            core_distances=model_OPTIC.core_distances_,
            ordering=model_OPTIC.ordering_, eps=eps_v)
        # Identify points that are *not* labeled as "noise" (labeled as -1)
        msk = labels_dbs != -1
        # Return data array with points labeled as noise filterd out
        return data[msk]

    # Points (stars) ordered by "reachability", i.e: "eps", which is the
    # "maximum distance between two samples for one to be considered as in the
    # neighborhood of the other".
    reachability = model_OPTIC.reachability_[model_OPTIC.ordering_]

    # Select the center of the cluster using the 1th percentile of stars with
    # the smallest reachability values
    eps_min = np.percentile(reachability, 1)
    data_msk = mskNoise(eps_min)
    # The mean of these points is considered to be the center of the cluster
    # we are looking for
    center = np.array([data_msk.mean(0)])

    # Go through (almost) all the 'eps' values, starting from the smallest
    # value larger that the minimum eps (+ eps_step).
    # At each step:
    # 1. request the mask filtering out the "noise", using 'eps_c'
    # 2. find the distance from these points ("members") to the defined center
    # 3. see if 'eps_c' fulfills the breaking condition. If it does, break out
    # of the block.
    for eps_c in np.arange(
            reachability.min() + eps_step,
            np.percentile(reachability, 95), eps_step):

        # Distance from the estimated members (i.e., data the noise) to the
        # center of the cluster (defined before this 'for' block).
        data_msk = mskNoise(eps_c)
        dist_c = distance.cdist(center, data_msk)

        # Maximum distance for any point in the data_msk array to the center
        max_dist = dist_c.max()
        # Percentile of the distances using the 'perc_cut' value
        perc_dist = np.percentile(dist_c, perc_cut)

        # print(eps_c, len(mskNoise(eps_c)), max_dist, np.percentile(dist_c, (20, 40, 60, 80)))

        # Breaking condition: if the difference between the maximum distance
        # and the percentile distance is larger than the eps_c value.
        # The idea is to use an 'eps' value where the member further away from
        # the defined cluster center is not that far away from the 'perc_cut'
        # percentile.
        # Thus, a smaller 'perc_cut' value will result in smaller
        # selected 'eps' values, i.e fewer members. This is because a smaller
        # 'perc_cut' is more restrictive
        if eps_c < max_dist - perc_dist:
            # print("selected eps:", eps_c)
            break

    return eps_c


def runOPTICS(data, min_samples):
    """
    min_samplesint: > 1 or float between 0 and 1 (default=5)
      The number of samples in a neighborhood for a point to be considered as
      a core point. Also, up and down steep regions can't have more then
      min_samples consecutive non-steep points. Expressed as an absolute
      number or a fraction of the number of samples (rounded to be at least 2).
    """
    model_OPTIC = skclust.OPTICS(min_samples=min_samples)
    # Fit the model
    model_OPTIC.fit(data)

    import matplotlib.pyplot as plt
    space = np.arange(len(data))
    reachability = model_OPTIC.reachability_[model_OPTIC.ordering_]
    labels = model_OPTIC.labels_[model_OPTIC.ordering_]
    # plt.plot(space[labels != -1], reachability[labels != -1], 'g.', alpha=0.7)
    plt.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.5)
    plt.show()

    return model_OPTIC


def readFiles():
    """
    Read files from the input folder
    """
    files = []
    for pp in Path('input').iterdir():
        if pp.is_file():
            files += [pp]

    return files


def dataExtract(file_path):
    """
    """
    # Read data from file
    print("{:<21}: {}".format("\nFile", file_path.parts[-1]))
    data_all = Table.read(
        file_path, format='ascii', fill_values=[('', '0'), ('nan', '0')])
    N_d = len(data_all)
    print("{:<20}: {}".format("Stars read", N_d))

    # Remove stars with no valid (ie: masked) data
    try:
        msk_accpt = np.logical_and.reduce([
            ~data_all[_].mask for _ in data_cols])
        data = data_all[msk_accpt]
        print("{:<20}: {}".format("Stars removed", N_d - len(data)))
    except AttributeError:
        # If there are no masked columns, then all data is valid
        msk_accpt = np.array([True for _ in range(N_d)])
        data = data_all

    # Extract ID, data, and its uncertainties
    data_id, data_c, data_err = data_all[id_col], data[data_cols],\
        data[err_cols]

    return data_all, data_id, data_c, data_err, msk_accpt


def reSampleData(data, data_err):
    """
    Re-sample the data given its uncertainties using a normal distribution
    """
    # Re-arrange into proper shape
    data_arr = np.array([data[_] for _ in data.columns]).T
    e_data_arr = np.array([data_err[_] for _ in data_err.columns]).T

    # Gaussian random sample
    grs = np.random.normal(0., 1., data_arr.shape[0])
    sampled_data = data_arr + grs[:, np.newaxis] * e_data_arr

    return sampled_data


def dimReduc(data, PCAdims=2):
    """
    Perform PCA and feature reduction
    """
    pca = PCA(n_components=PCAdims)
    data_pca = pca.fit(data).transform(data)
    # print("Selected N={} PCA features".format(PCAdims))
    # var_r = ["{:.2f}".format(_) for _ in pca.explained_variance_ratio_]
    # print(" Variance ratio: ", ", ".join(var_r))

    return data_pca


if __name__ == '__main__':
    main()
