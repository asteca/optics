
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
from astropy.table import Table
from astropy.io import ascii
from clustering import perc_cut, readFiles, dataExtract


prob_cut = 0.5


def main():
    """
    """
    files = readFiles()
    for file_path in files:
        fname = file_path.parts[-1].split('.')[0]
        data_all = dataExtract(file_path)[0]

        fname, fext = file_path.parts[-1].split('.')
        fprobs = 'output/' + fname + '_probs.' + fext
        probs_mean = ascii.read(fprobs, header_start=0)['probs_mean']

        makePlot(data_all, probs_mean, fname)


def makePlot(data_all, probs_mean, fname, Plx_offset=0.029):
    """
    Make plots using the final probabilities in the "_probs.dat" files.
    """
    # prob_cut = min(prob_cut, probs_mean.max())
    msk_memb = probs_mean >= min(prob_cut, probs_mean.max())

    fig = plt.figure(figsize=(15, 10))
    G = gridspec.GridSpec(4, 3)
    ax1 = plt.subplot(G[0:2, 0])
    ax21 = plt.subplot(G[0:1, 1])
    ax22 = plt.subplot(G[1:2, 1])
    ax3 = plt.subplot(G[0:2, 2])
    ax4 = plt.subplot(G[2:4, 0])
    ax5 = plt.subplot(G[2:4, 1])
    ax6 = plt.subplot(G[2:4, 2])

    plt.suptitle("Percentile distance used to estimate the 'eps': {}".format(
        perc_cut))

    data_all['Plx'] += Plx_offset

    # # Reachability plot
    # space = np.arange(len(data))
    # msk = reachability < eps
    # ax1.plot(space[~msk], reachability[~msk], 'k.', alpha=0.3)
    # ax1.plot(space[msk], reachability[msk], 'g.')
    # ax1.axhline(eps, c='k', ls='-.', label="eps={:.3f}".format(eps))
    # ax1.set_ylabel('Reachability (epsilon distance)')
    # ax1.set_title('Reachability Plot (min_samples={})'.format(
    #     min_samples))
    # ymin, ymax = max(0, eps - eps * .5), min(3. * eps, max(reachability))
    # ax1.set_xlim(0, max(space[reachability < ymax]))
    # ax1.set_ylim(ymin, ymax)
    # ax1.legend()

    ax1.hist(probs_mean, alpha=.5)
    ax1.axvline(prob_cut, c='g', zorder=6)
    ax12 = ax1.twinx()
    ax12.yaxis.set_minor_locator(AutoMinorLocator(4))
    xx, yy = np.linspace(probs_mean.min(), probs_mean.max(), 100), []
    for pr in xx:
        yy.append((probs_mean >= pr).sum())
    ax12.plot(xx, yy, c='r', lw=3, label=r"$P_{{cut}}$={:.2f}".format(
        prob_cut))
    ax12.set_ylabel(r'$N_{stars}$')
    # ax12.set_yscale('log')
    # ax12.set_yticks([])
    ax1.set_xlabel('Probabilities')
    ax1.set_ylabel(r'$\log (N)$')
    ax1.set_yscale('log')
    plt.legend()

    plx_prob, pmRA_prob, pmDE_prob = [], [], []
    for pp in np.arange(.05, .95, .01):
        msk = probs_mean >= pp
        plx_prob.append([
            pp, data_all['Plx'][msk].mean(), data_all['Plx'][msk].std()])
        pmRA_prob.append([
            pp, data_all['pmRA'][msk].mean(), data_all['pmRA'][msk].std()])
        pmDE_prob.append([
            pp, data_all['pmDE'][msk].mean(), data_all['pmDE'][msk].std()])

    plx_prob, pmRA_prob, pmDE_prob = [
        np.array(_).T for _ in (plx_prob, pmRA_prob, pmDE_prob)]

    ax21.plot(pmDE_prob[0], pmDE_prob[1], c='b', label='pmDE')
    ax21.fill_between(
        pmDE_prob[0], pmDE_prob[1] - pmDE_prob[2], pmDE_prob[1] + pmDE_prob[2],
        color='blue', alpha=0.1)
    ax21.axvline(prob_cut, c='g', zorder=6)
    ax21.set_xlabel(r'$P_{cut}$')
    ax21.set_ylabel(r'$\mu_{\delta}$ [mas/yr]')

    ax22.plot(pmRA_prob[0], pmRA_prob[1], c='r', label='pmRA')
    ax22.fill_between(
        pmRA_prob[0], pmRA_prob[1] - pmRA_prob[2], pmRA_prob[1] + pmRA_prob[2],
        color='red', alpha=0.1)
    ax22.axvline(prob_cut, c='g', zorder=6)
    ax22.set_ylabel(r'$\mu_{\alpha} \cos \delta$ [mas/yr]')
    ax22.set_xlabel(r'$P_{cut}$')

    ax3.plot(plx_prob[0], plx_prob[1])
    ax3.fill_between(
        plx_prob[0], plx_prob[1] - plx_prob[2], plx_prob[1] + plx_prob[2],
        color='k', alpha=0.1)
    ax3.axvline(prob_cut, c='g', zorder=6)
    ax3.set_ylabel(r'$Plx$')
    ax3.set_xlabel(r'$P_{cut}$')

    ax4.set_title("N={}".format(len(data_all['RA_ICRS'][msk_memb])))
    ax4.scatter(
        data_all['RA_ICRS'][msk_memb], data_all['DE_ICRS'][msk_memb],
        marker='o', edgecolor='w', lw=.3, zorder=5)
    ax4.plot(
        data_all['RA_ICRS'][~msk_memb],
        data_all['DE_ICRS'][~msk_memb], 'k.', alpha=0.3, zorder=1)
    ax4.set_xlabel('RA')
    ax4.set_ylabel('DE')
    ax4.set_xlim(max(data_all['RA_ICRS']), min(data_all['RA_ICRS']))
    ax4.set_ylim(min(data_all['DE_ICRS']), max(data_all['DE_ICRS']))

    # PMs plot
    pmRA_mean = (
        data_all['pmRA'][msk_memb] /
        np.cos(np.deg2rad(data_all['pmDE'][msk_memb]))).mean()
    pmDE_mean = data_all['pmDE'][msk_memb].mean()
    ax5.set_title(r"$(\mu_{\alpha}, \mu_{\delta})=$" +
                  "({:.3f}, {:.3f})".format(pmRA_mean, pmDE_mean))
    ax5.scatter(
        data_all['pmRA'][msk_memb], data_all['pmDE'][msk_memb], marker='.',
        edgecolor='w', lw=.1, alpha=.7, zorder=5)
    ax5.plot(
        data_all['pmRA'][~msk_memb], data_all['pmDE'][~msk_memb],
        'k+', alpha=0.3, zorder=1)
    ax5.set_xlabel(r'$\mu_{\alpha} \cos \delta$ [mas/yr]')
    ax5.set_ylabel(r'$\mu_{\delta}$ [mas/yr]')
    xmin, xmax = np.percentile(data_all['pmRA'][~msk_memb], (25, 75))
    ymin, ymax = np.percentile(data_all['pmDE'][~msk_memb], (25, 75))
    xclmin = data_all['pmRA'][msk_memb].mean() -\
        5. + data_all['pmRA'][msk_memb].std()
    xclmax = data_all['pmRA'][msk_memb].mean() +\
        5. + data_all['pmRA'][msk_memb].std()
    yclmin = data_all['pmDE'][msk_memb].mean() -\
        5. + data_all['pmDE'][msk_memb].std()
    yclmax = data_all['pmDE'][msk_memb].mean() +\
        5. + data_all['pmDE'][msk_memb].std()
    ax5.set_xlim(max(xmax, xclmax), min(xmin, xclmin))
    ax5.set_ylim(min(ymin, yclmin), max(ymax, yclmax))

    # Plxs plot
    ax6.set_title("Plx offset: +{}".format(Plx_offset))
    xmin = np.percentile(data_all['Plx'][msk_memb], 1) -\
        3. * data_all['Plx'][msk_memb].std()
    xmax = np.percentile(data_all['Plx'][msk_memb], 95) +\
        3. * data_all['Plx'][msk_memb].std()
    msk1 = np.logical_and.reduce([
        (data_all['Plx'] > -2), (data_all['Plx'] < 4), (msk_memb)])
    msk2 = np.logical_and.reduce([
        (data_all['Plx'] > -2), (data_all['Plx'] < 4), (~msk_memb)])
    ax6.hist(data_all['Plx'][msk1], 20, density=True, alpha=.7, zorder=5)
    ax6.hist(
        data_all['Plx'][msk2], 75, color='k', alpha=0.3, density=True,
        zorder=1)
    plx_mean = data_all['Plx'][msk_memb].mean()
    plx_16, plx_84 = np.percentile(data_all['Plx'][msk_memb], (16, 84))
    ax6.axvline(
        plx_mean, c='r',
        label=r"$Plx_{{mean}}={:.3f}_{{{:.3f}}}^{{{:.3f}}}$".format(
            plx_mean, plx_16, plx_84), zorder=6)
    ax6.axvline(plx_16, c='orange', ls='--', zorder=6)
    ax6.axvline(plx_84, c='orange', ls='--', zorder=6)
    ax6.set_xlabel('Plx')
    ax6.set_xlim(xmin, xmax)
    ax6.legend()

    file_out = 'output/' + fname + '.png'
    fig.tight_layout()
    plt.savefig(file_out, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
