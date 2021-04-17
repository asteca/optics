# optics

This method uses the OPTICS[1] algorithm to estimate membership probabilities
when the cluster occupies most of the observed frame, and thus there is no
identifiable overdensity in the (x, y) coordinates space.

Both pyUPMASK and the Bayesian method in ASteCA require that the cluster
should show an overdensity in the coordinates space, and hence are not
appropriate for the cases handled by this method.

The OPTICS algorithm in this method works on the parallax and proper motions
space, looking for stars in regions of overdensity. Hence, no field region
in coordinates space is required at all.

Becomes slow for large fields.

> OPTICS: Ordering Points To Identify the Clustering Structure Mihael Ankerst, Markus M. Breunig, Hans-Peter Kriegel, Jörg Sander (1999): "..the reachability-plot is rather insensitive to the input parameters of the method, i.e. the generating distance \epsilon and the value for MinPts. Roughly speaking, the values have just to be "large" enough to yield a good result. The concrete values are not crucial because there is a broad range of possible values for which we always can see the clustering structure of a data set when looking at the corresponding reachability-plot."

[1]:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html