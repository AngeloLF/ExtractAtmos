import numpy as np
import matplotlib.pyplot as plt
import sys, json, os, shutil
import coloralf as c
# import alftool as alf
from scipy import interpolate
from time import time
from copy import deepcopy

sys.path.append('./Spec2vecModels/')
from get_argv import get_argv

# IMPORTATION SPECTRACTOR
spectractor_version = "Spectractor" 
for argv in sys.argv:
    if "=" in argv and argv.split("=")[0] == "specver":
        spectractor_version = argv.split("=")[1]
sys.path.append(f"./{spectractor_version}")
from spectractor import parameters
from spectractor.extractor.spectrum import Spectrum
from spectractor.fit.fit_spectrogram import SpectrogramFitWorkspace, run_spectrogram_minimisation
from spectractor.fit.fit_spectrum import SpectrumFitWorkspace, run_spectrum_minimisation






def showHeader(header):

    keys = list(header.keys())
    print(f"\nHEADER : ")

    for k in keys:
        print(f"{c.lr}{k}{c.d} : {header[k]}")






def recupAtmosFromParams(w, file_json, wanted_labels=["vaod", "ozone", "pwv", "d_ccd"]):

    data = dict()

    for l, v, e in zip(w.params.labels, w.params.values, w.params.err):
        
        for wl in wanted_labels:
            if wl.lower() in l.lower():
                data[wl] = [float(v), float(e)]

    with open(file_json, "w") as f:
        json.dump(data, f)






def extractOne(Args, num_str, path="./results/output_simu", atmoParamFolder="atmos_params_fit"):

    parameters.DISPLAY = False
    debug = False

    ### Importation hparams & variable params
    with open(f"{path}/{Args.test}/hparams.json", "r") as fjson:
        hp = json.load(fjson)

    if Args.model == "true":
        predFolder = "spectrum"
        saveFolder = "true"

    elif Args.model == "spectractorfile":
        predFolder = "Spectractor"
        saveFolder = "Spectractor"

    else:
        predFolder = f"pred_{Args.fullname}"
        saveFolder = f"pred_{Args.fullname}"

    file_json = f"{path}/{Args.test}/{atmoParamFolder}/{saveFolder}/atmos_params_{num_str}_spectrum.json"
    if atmoParamFolder not in os.listdir(f"{path}/{Args.test}"):
        try:
            os.mkdir(f"{path}/{Args.test}/{atmoParamFolder}")
        except:
            print(f"WARNING [extractAtmos.py] : mkdir of {path}/{Args.test}/{atmoParamFolder} not work")
    if saveFolder not in os.listdir(f"{path}/{Args.test}/{atmoParamFolder}"):
        try:
            os.mkdir(f"{path}/{Args.test}/{atmoParamFolder}/{saveFolder}")
        except:
            print(f"WARNING [extractAtmos.py] : mkdir of {path}/{Args.test}/{atmoParamFolder}/{saveFolder} not work")


    ### EXTRACTION with the spectrum
    c.fg(f"INFO [extractAtmo.py] : Begin Spectrum Minimisation for {Args.test}/{predFolder}/spectrum_{num_str}.npy ...")
    if f"images_{num_str}_spectrum.fits" in os.listdir(f"{path}/{Args.test}/spectrum_fits"):
        file_name = f"{path}/{Args.test}/spectrum_fits/images_{num_str}_spectrum.fits"
        spec = Spectrum(file_name, fast_load=True)

        # showHeader(spec.header)
        print(f"Size of lambdas : {np.size(spec.data)}")

        if "debug" in sys.argv:
            parameters.DEBUG = True
            parameters.VERBOSE = True
            parameters.DISPLAY = True # oskour ...
            debug = True

        if predFolder is not None and predFolder != "Spectractor": # need to change data / lambdas_binw / err / cov_matrix

            if not "Spectractor" in predFolder:
                spec.header["D2CCD"] = hp["DISTANCE2CCD"]
                print(f"Change d2ccd")

            spec.convert_from_flam_to_ADUrate()
            x = np.arange(300, 1100).astype(float)
            y = np.load(f"{path}/{Args.test}/{predFolder}/spectrum_{num_str}.npy")
            finterp = interpolate.interp1d(x, y, kind='linear', bounds_error=False, fill_value=0.0)
            spec.data = finterp(spec.lambdas) / spec.gain / spec.expo
            spec.convert_from_ADUrate_to_flam()

        w = SpectrumFitWorkspace(spec, atmgrid_file_name="", verbose=debug, plot=debug, live_fit=False, fit_angstrom_exponent=True)
        w.filename = ""
        if "date_obs" not in dir(w.spectrum):
            w.spectrum.date_obs = "2017-05-31T02:53:52.356"
            print(f"Info [extractAtmos.py] : DATE-OBS not given")
        run_spectrum_minimisation(w, method="newton")
        recupAtmosFromParams(w, file_json)
    else:
        print(f"Info [extractAtmos.py] : fits {path}/{Args.test}/spectrum_fits/images_{num_str}_spectrum.fits not exist [skip this one]")








def getTrueValues(hp, vp, label):

    if label == "vaod" : return vp["ATM_AEROSOLS"]
    elif label == "ozone" : return vp["ATM_OZONE"]
    elif label == "pwv" : return vp["ATM_PWV"]
    elif label == "d_ccd" : return hp["DISTANCE2CCD"]
    else : raise Exception(f"In [extractAtmo.py/getTrueValues] : label {label} unknow")



def analyseExtraction(Args, path="./results/output_simu", atmoParamFolder="atmos_params_fit", atmoParamFolderSave="atmos_params_figure", colors=["r", "g", "b", "y", "m"]):

    targets = ["vaod", "ozone", "pwv", "d_ccd"]
    nums_str = np.sort([fspectrum.split("_")[1][:-4] for fspectrum in os.listdir(f"{path}/{Args.test}/spectrum")])

    if atmoParamFolderSave in os.listdir(f"{path}/{Args.test}"):
        shutil.rmtree(f"{path}/{Args.test}/{atmoParamFolderSave}")
    os.mkdir(f"{path}/{Args.test}/{atmoParamFolderSave}")

    for t in targets:
        os.mkdir(f"{path}/{Args.test}/{atmoParamFolderSave}/{t}")

    
    saveFolders = [pf for pf in os.listdir(f"{path}/{Args.test}/{atmoParamFolder}") if not "." in pf]
    saveFolders_str = list()

    full_data = dict() # {savef : {t:[list(), list()] for t in targets} for savef in saveFolders}

    for savef in saveFolders:

        if savef.startswith("pred_"):
            saveFolders_str.append("_".join(savef.split("_")[1:3]))
        else:
            saveFolders_str.append(savef)

        rdata = {t:[np.zeros(len(nums_str)), np.zeros(len(nums_str))] for t in targets}

        for i, n in enumerate(nums_str):

            if f"atmos_params_{n}_spectrum.json" in os.listdir(f"{path}/{Args.test}/{atmoParamFolder}/{savef}"):

                with open(f"{path}/{Args.test}/{atmoParamFolder}/{savef}/atmos_params_{n}_spectrum.json", "r") as f:

                    data = json.load(f)

                for t in targets:
                    rdata[t][0][i] = data[t][0]
                    rdata[t][1][i] = data[t][1]

            else:

                print(f"Info [extractAtmos.py] in analyse, skip atmos_params_{n}_spectrum.json")
                for t in targets:
                    rdata[t][0][i] = np.nan
                    rdata[t][1][i] = np.nan

        full_data[savef] = deepcopy(rdata)



    ### Importation hparams & variable params
    with open(f"{path}/{Args.test}/hparams.json", "r") as fjson:
        hp = json.load(fjson)
    vp = np.load(f"{path}/{Args.test}/vparams.npz")

    save_txt = "Save extract atmo performances :\n"
    print("\nSave extract atmo performances :")

    scores = {savef:dict() for savef in saveFolders}

    for i, t in enumerate(targets):

        save_txt += f"\n{t}\n"
        print(f"\n{t}")

        true_vals = getTrueValues(hp, vp, t)
        if t in ["ozone", "vaod", "pwv"]:
            true_sort = np.argsort(true_vals)
            x = true_vals[true_sort]
            y = true_vals[true_sort]
        else:
            true_sort = np.arange(len(nums_str))
            x = np.arange(len(nums_str))
            y = true_vals


        for mode in ["plot", "subplot", "full"]:

            plt.figure(figsize=(16, 9))

            for i, savef in enumerate(saveFolders):

                # color
                if savef == "true":
                    color = "k"
                elif i < len(colors):
                    color = colors[i]
                else:
                    color = None

                res = full_data[savef][t][0][true_sort]-y
                
                score = np.nanmean(np.abs(res))
                std = np.nanstd(np.abs(res))
                score_mean = np.nanmean(res)
                score_std = np.nanstd(res)

                if mode == "plot":
                    save_txt += f"{savef} : {score:.3f} +- {std:.3f} --- {score_mean:.3f} +- {score_std:.3f}\n"
                    print(f"{savef} : {score:.3f} +- {std:.3f} --- {score_mean:.3f} +- {score_std:.3f}")
                    scores[savef][t] = [score, std]

                if mode != "full":
                    if mode == "subplot" : plt.subplot(2, 2, i+1)
                    plt.errorbar(x, res, yerr=full_data[savef][t][1][true_sort], color=color, ls="", marker=".", label=f"{savef} : {score:.3f}")
                    plt.plot()
                    plt.xlabel(t)
                    plt.ylabel("Residus")
                    plt.axhline(0, color="k", ls=":", label="True value")
                    plt.title(f"{savef} : residus abs = {score:.3f}$\pm${std:.3f} [mean={score_mean:.3f}$\pm${score_std:.3f}]")
                    plt.ylim(np.nanmin(res), np.nanmax(res))
                    if mode == "plot": 
                        plt.savefig(f"{path}/{Args.test}/{atmoParamFolderSave}/{t}/{t}_{savef}.png")
                        plt.close()
                else:
                    plt.plot(x, res, color=color, ls="", marker=".")

            if mode == "full":
                plt.xlabel(t)
                plt.ylabel("Residus")
                plt.axhline(0, color="k", ls=":", label="True value")
                plt.legend()
            
                plt.tight_layout()
                plt.savefig(f"{path}/{Args.test}/{atmoParamFolderSave}/full_{t}.png")
                plt.close()

            elif mode == "subplot":
                plt.tight_layout()
                plt.savefig(f"{path}/{Args.test}/{atmoParamFolderSave}/subplot_{t}.png")
                plt.close()

    with open(f"{path}/{Args.test}/{atmoParamFolderSave}/save_extraction_score.txt", "w") as f:
        f.write(save_txt)

    borne_PWV = hp["vparams"]["ATM_PWV"]
    borne_VAOD = hp["vparams"]["ATM_AEROSOLS"]
    borne_OZONE = hp["vparams"]["ATM_OZONE"]

    for savef, vals in scores.items():

        o, v, p = vals["ozone"], vals["vaod"], vals["pwv"]
        scores[savef]["total"] = [
            (o[0] / (borne_OZONE[1] - borne_OZONE[0]) + v[0] / (borne_VAOD[1] - borne_VAOD[0]) + p[0] / (borne_PWV[1] - borne_PWV[0])) * 100., # en %
            np.sqrt((o[1] / (borne_OZONE[1] - borne_OZONE[0]))**2 + (v[1] / (borne_VAOD[1] - borne_VAOD[0]))**2 + (p[1] / (borne_PWV[1] - borne_PWV[0]))**2) * 100., # en %
        ]


    for inPC in [False, True]:

        plt.figure(figsize=(16, 9))

        for i, (t, borne) in enumerate(zip(["pwv", "vaod", "ozone", "total"], [borne_PWV, borne_VAOD, borne_OZONE, None])):

            x = np.arange(len(saveFolders))
            divide = (borne[1] - borne[0])/100. if inPC and borne is not None else 1.0
            y = [scores[savef][t][0] / divide for savef in saveFolders]
            yerr = [scores[savef][t][1] / divide for savef in saveFolders]

            plt.subplot(2, 2, i+1)
            plt.errorbar(x, y, yerr=yerr, color=colors[i], ls="", marker=".")
            plt.xticks(x, saveFolders)
            if inPC or t == "total":
                plt.ylabel(f"{t} (%)")
            else:
                plt.ylabel(f"{t}")

        plt.tight_layout()
        if inPC:
            plt.savefig(f"{path}/{Args.test}/{atmoParamFolderSave}/resume_all_INPC.png")
        else:
            plt.savefig(f"{path}/{Args.test}/{atmoParamFolderSave}/resume_all.png")







if __name__ == "__main__":

    """
    For extract atmos with spectractor minimisation !
    From pred spectrum or :
        * true : using true spectrum from simulation
        * spectractorfile : using spectrum.fits product of spectractor extraction (different from pred_Spectractor_x_x_0e+00 -> 2 interpolation)
    """

    # arguments needed
    path = "./results/output_simu"
    atmoParamFolder = "atmos_params_fit"
    atmoParamFolderSave = "atmos_params_figure"

    if "extract_atmo" in sys.argv: 
        Args = get_argv(sys.argv[1:], prog="extract_atmo")

        # multiple cpu ?
        nrange = None
        for arg in sys.argv[1:]:
            if arg[:6] == "range=" : nrange = arg[6:]

        nums_str = np.sort([fspectrum.split("_")[1][:-4] for fspectrum in os.listdir(f"{path}/{Args.test}/spectrum")])

        # build partition
        if nrange is None:
            partition = [None]*len(nums_str)
        else:
            nbegin, nsimu = nrange.split("_")
            partition = np.arange(int(nbegin), int(nbegin) + int(nsimu))

        t0 = time()
        
        for n in nums_str:

            if partition[0] is None or int(n) in partition:

                extractOne(Args, n, path=path, atmoParamFolder=atmoParamFolder)

        tf = time() - t0
        print(f"Finish in {tf/60:.1f} min [{tf/len(partition):.1f} sec/extraction]")

    elif "analyse_atmo" in sys.argv:

        Args = get_argv(sys.argv[1:], prog="analyse_atmo")

        analyseExtraction(Args, path, atmoParamFolder, atmoParamFolderSave)

    else:

        print("No argv know (need extract_atmo or analyse_atmo)")





