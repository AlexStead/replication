import sys
import fronpy
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import time
import scipy

electricity = fronpy.dataset('electricity.csv')
electricitygb = fronpy.dataset('electricitygb_recentred.csv')
air = fronpy.dataset('airlines.csv')
rice = fronpy.dataset('philippines_ricetl.csv')
rail = fronpy.dataset('swissrail.csv')

tic = time.perf_counter()
nexpmodel_rice = fronpy.estimate(rice,model='nexp')
toc = time.perf_counter()
nexpmodel_rice_time = toc-tic
tic = time.perf_counter()
ngmodel_rice = fronpy.estimate(rice,model='ng',startingvalues=np.append(nexpmodel_rice.theta,0))
toc = time.perf_counter()
ngmodel_rice_time = toc-tic
tic = time.perf_counter()
nhnmodel_rice = fronpy.estimate(rice,model='nhn')
toc = time.perf_counter()
nhnmodel_rice_time = toc-tic
tic = time.perf_counter()
ntnmodel_rice = fronpy.estimate(rice,model='ntn',startingvalues=np.append(nhnmodel_rice.theta,0))
toc = time.perf_counter()
ntnmodel_rice_time = toc-tic
tic = time.perf_counter()
nnmodel_rice = fronpy.estimate(rice,model='nn',startingvalues=np.append(nhnmodel_rice.theta,np.log(0.5)))
nnmodel_rice = fronpy.estimate(rice,model='nn',startingvalues=np.append(nhnmodel_rice.theta,np.log(0.5)))
toc = time.perf_counter()
nnmodel_rice_time = toc-tic

tic = time.perf_counter()
nexpmodel_electricity = fronpy.estimate(electricity,model='nexp',cost=True)
toc = time.perf_counter()
nexpmodel_electricity_time = toc-tic
tic = time.perf_counter()
ngmodel_electricity = fronpy.estimate(electricity,model='ng',startingvalues=np.append(nexpmodel_electricity.theta,0),cost=True)
toc = time.perf_counter()
ngmodel_electricity_time = toc-tic
tic = time.perf_counter()
nhnmodel_electricity = fronpy.estimate(electricity,model='nhn',cost=True)
toc = time.perf_counter()
nhnmodel_electricity_time = toc-tic
tic = time.perf_counter()
ntnmodel_electricity = fronpy.estimate(electricity,model='ntn',startingvalues=np.append(nhnmodel_electricity.theta,0),cost=True)
toc = time.perf_counter()
ntnmodel_electricity_time = toc-tic
tic = time.perf_counter()
nnmodel_electricity = fronpy.estimate(electricity,model='nn',startingvalues=np.append(nhnmodel_electricity.theta,np.log(0.5)),cost=True)
toc = time.perf_counter()
nnmodel_electricity_time = toc-tic
tic = time.perf_counter()

tic = time.perf_counter()
nexpmodel_electricitygb = fronpy.estimate(electricitygb,model='nexp',cost=True)
toc = time.perf_counter()
nexpmodel_electricitygb_time = toc-tic
tic = time.perf_counter()
nhnmodel_electricitygb = fronpy.estimate(electricitygb,model='nhn',cost=True)
toc = time.perf_counter()
nhnmodel_electricitygb_time = toc-tic
tic = time.perf_counter()
ntnmodel_electricitygb = fronpy.estimate(electricitygb,model='ntn',cost=True)
toc = time.perf_counter()
ntnmodel_electricitygb_time = toc-tic
tic = time.perf_counter()
nnmodel_electricitygb = fronpy.estimate(electricitygb,model='nn',startingvalues=np.append(nhnmodel_electricitygb.theta,np.log(0.5)),cost=True)
toc = time.perf_counter()
nnmodel_electricitygb_time = toc-tic
tic = time.perf_counter()
tic = time.perf_counter()
ngmodel_electricitygb = fronpy.estimate(electricitygb,model='ng',startingvalues=np.append(nexpmodel_electricitygb.theta,np.log(1)),cost=True)
toc = time.perf_counter()
ngmodel_electricitygb_time = toc-tic

predictions = {'nexp_electricity_resid': nexpmodel_electricity.residual,
               'nexp_electricity_eff': nexpmodel_electricity.eff_bc,
               'ng_electricity_resid': ngmodel_electricity.residual,
               'ng_electricity_eff': ngmodel_electricity.eff_bc,
               'nhn_electricity_resid': nhnmodel_electricity.residual,
               'nhn_electricity_eff': nhnmodel_electricity.eff_bc,
               'nn_electricity_resid': nnmodel_electricity.residual,
               'nn_electricity_eff': nnmodel_electricity.eff_bc,
               'ntn_electricity_resid': ntnmodel_electricity.residual,
               'ntn_electricity_eff': ntnmodel_electricity.eff_bc}
predictions = pd.DataFrame(predictions)
predictions.to_csv('predictions_electricity.csv', index=False)
predictions = {'nexp_rice_resid': nexpmodel_rice.residual,
               'nexp_rice_eff': nexpmodel_rice.eff_bc,
               'ng_rice_resid': ngmodel_rice.residual,
               'ng_rice_eff': ngmodel_rice.eff_bc,
               'nhn_rice_resid': nhnmodel_rice.residual,
               'nhn_rice_eff': nhnmodel_rice.eff_bc,
               'nn_rice_resid': nnmodel_rice.residual,
               'nn_rice_eff': nnmodel_rice.eff_bc,
               'ntn_rice_resid': ntnmodel_rice.residual,
               'ntn_rice_eff': ntnmodel_rice.eff_bc}
predictions = pd.DataFrame(predictions)
predictions.to_csv('predictions_rice.csv', index=False)
predictions = {'nexp_electricitygb_resid': nexpmodel_electricitygb.residual,
               'nexp_electricitygb_eff': nexpmodel_electricitygb.eff_bc,
               'ng_electricitygb_resid': ngmodel_electricitygb.residual,
               'ng_electricitygb_eff': ngmodel_electricitygb.eff_bc,
               'nhn_electricitygb_resid': nhnmodel_electricitygb.residual,
               'nhn_electricitygb_eff': nhnmodel_electricitygb.eff_bc,
               'nn_electricitygb_resid': nnmodel_electricitygb.residual,
               'nn_electricitygb_eff': nnmodel_electricitygb.eff_bc,
               'ntn_electricitygb_resid': ntnmodel_electricitygb.residual,
               'ntn_electricitygb_eff': ntnmodel_electricitygb.eff_bc}
predictions = pd.DataFrame(predictions)
predictions.to_csv('predictions_electricitygb.csv', index=False)

mineff = min(np.concatenate((nexpmodel_electricity.eff_bc,
                             ngmodel_electricity.eff_bc,
                             nhnmodel_electricity.eff_bc,
                             nnmodel_electricity.eff_bc,
                             ntnmodel_electricity.eff_bc)))
points_electricity = np.linspace(mineff, 1, num=1000)
nexp_electricity_density = gaussian_kde(nexpmodel_electricity.eff_bc,bw_method='scott').evaluate(points_electricity)
ng_electricity_density = gaussian_kde(ngmodel_electricity.eff_bc,bw_method='scott').evaluate(points_electricity)
nhn_electricity_density = gaussian_kde(nhnmodel_electricity.eff_bc,bw_method='scott').evaluate(points_electricity)
nn_electricity_density = gaussian_kde(nnmodel_electricity.eff_bc,bw_method='scott').evaluate(points_electricity)
ntn_electricity_density = gaussian_kde(ntnmodel_electricity.eff_bc,bw_method='scott').evaluate(points_electricity)

kerneldensities = {'points_electricity': points_electricity,
                   'nexp_electricity_density': nexp_electricity_density,
                   'ng_electricity_density': ng_electricity_density,
                   'nhn_electricity_density': nhn_electricity_density,
                   'nn_electricity_density': nn_electricity_density,                  
                   'ntn_electricity_density': ntn_electricity_density}
kerneldensities = pd.DataFrame(kerneldensities)
kerneldensities.to_csv('kerneldensities_electricity.csv', index=False)

mineff = min(np.concatenate((nexpmodel_rice.eff_bc,
                             ngmodel_rice.eff_bc,
                             nhnmodel_rice.eff_bc,
                             nnmodel_rice.eff_bc,
                             ntnmodel_rice.eff_bc)))
points_rice = np.linspace(mineff, 1, num=1000)
nexp_rice_density = gaussian_kde(nexpmodel_rice.eff_bc,bw_method='scott').evaluate(points_rice)
ng_rice_density = gaussian_kde(ngmodel_rice.eff_bc,bw_method='scott').evaluate(points_rice)
nhn_rice_density = gaussian_kde(nhnmodel_rice.eff_bc,bw_method='scott').evaluate(points_rice)
nn_rice_density = gaussian_kde(nnmodel_rice.eff_bc,bw_method='scott').evaluate(points_rice)
ntn_rice_density = gaussian_kde(ntnmodel_rice.eff_bc,bw_method='scott').evaluate(points_rice)

kerneldensities = {'points_rice': points_rice,
                   'nexp_rice_density': nexp_rice_density,
                   'ng_rice_density': ng_rice_density,
                   'nhn_rice_density': nhn_rice_density,
                   'nn_rice_density': nn_rice_density,
                   'ntn_rice_density': ntn_rice_density,}
kerneldensities = pd.DataFrame(kerneldensities)
kerneldensities.to_csv('kerneldensities_rice.csv', index=False)

mineff = min(np.concatenate((nexpmodel_electricitygb.eff_bc,
                             ngmodel_electricitygb.eff_bc,
                             nhnmodel_electricitygb.eff_bc,
                             nnmodel_electricitygb.eff_bc,
                             ntnmodel_electricitygb.eff_bc)))
points_electricitygb = np.linspace(mineff, 1, num=1000)
nexp_electricitygb_density = gaussian_kde(nexpmodel_electricitygb.eff_bc,bw_method='scott').evaluate(points_electricitygb)
ng_electricitygb_density = gaussian_kde(ngmodel_electricitygb.eff_bc,bw_method='scott').evaluate(points_electricitygb)
nhn_electricitygb_density = gaussian_kde(nhnmodel_electricitygb.eff_bc,bw_method='scott').evaluate(points_electricitygb)
nn_electricitygb_density = gaussian_kde(nnmodel_electricitygb.eff_bc,bw_method='scott').evaluate(points_electricitygb)
ntn_electricitygb_density = gaussian_kde(ntnmodel_electricitygb.eff_bc,bw_method='scott').evaluate(points_electricitygb)

kerneldensities = {'points_electricitygb': points_electricitygb,
                   'nexp_electricitygb_density': nexp_electricitygb_density,
                   'ng_electricitygb_density': ng_electricitygb_density,
                   'nhn_electricitygb_density': nhn_electricitygb_density,
                   'nn_electricitygb_density': nn_electricitygb_density,                  
                   'ntn_electricitygb_density': ntn_electricitygb_density}
kerneldensities = pd.DataFrame(kerneldensities)
kerneldensities.to_csv('kerneldensities_electricitygb.csv', index=False)

coefs = np.chararray((5,4))
def coef(model, j):
    coefficient = model.theta[j]
    string = format(coefficient,'.4f')
    return string
def sigmav(model):
    coefficient = model.sigmav
    string = format(coefficient, '.4f')
    return string
def sigmau(model):
    coefficient = model.sigmau
    string = format(coefficient,'.4f')
    return string
def mu(model):
    coefficient = model.mu
    string = format(coefficient,'.4f')
    return string
def se(model, j):
    coefficient = model.theta_se[j]
    string = "(" + format(coefficient,'.4f') + ")"
    return string
def sigmavse(model):
    coefficient = model.sigmav_se
    string = "(" + format(coefficient,'.4f') + ")"
    return string
def sigmause(model):
    coefficient = model.sigmau_se
    string = "(" + format(coefficient,'.4f') + ")"
    return string
def muse(model):
    coefficient = model.mu_se
    string = "(" + format(coefficient,'.4f') + ")"
    return string
def stars(model,j):
        return model.theta_star[j]
def coefstars(model,j):
    string = coef(model,j) + "\\sym{" + stars(model,j) + "}"
    return string
def lnL(model):
    lnL = format(model.lnlikelihood,'.4f')
    return lnL
def LR(alternative,null):
    LR = 2*(alternative.lnlikelihood - null.lnlikelihood)
    if LR >= 10.828:
        stars = "***"
    elif LR >= 6.635:
        stars = "**"
    elif LR >= 3.841:
        stars = "*"
    else:
        stars = ""
    string = format(LR,'.4f') + "\\sym{" + stars + "}"
    return string


nexpmodel = nexpmodel_electricity
ngmodel = ngmodel_electricity
nhnmodel = nhnmodel_electricity
nnmodel = nnmodel_electricity
ntnmodel = ntnmodel_electricity

print("\\begin{table}[htbp]\\centering")
print("\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}")
print("\\caption{Parameter estimates -- Application 1}")
print("{\\renewcommand{\\arraystretch}{0.5}%")
print("\\begin{tabular}{l*{5}{D{.}{.}{8}}}")
print("\\toprule")
print(" & \\multicolumn{1}{c}{N-EXP} & \\multicolumn{1}{c}{N-G} & \multicolumn{1}{c}{N-HN} & \multicolumn{1}{c}{N-N} & \multicolumn{1}{c}{N-TN} \\\\")
print("\\midrule")
print("$\\beta_0$ & "
      + coefstars(nexpmodel,0) + " & "
      + coefstars(ngmodel,0) + " & "
      + coefstars(nhnmodel,0) + " & "
      + coefstars(nnmodel,0) + " & "
      + coefstars(ntnmodel,0)
      + " \\\\")
print(" & "
      + se(nexpmodel,0) + " & "
      + se(ngmodel,0) + " & "
      + se(nhnmodel,0) + " & "
      + se(nnmodel,0) + " & "
      + se(ntnmodel,0)
      + " \\\\")
print("$\\beta_1$ & "
      + coefstars(nexpmodel,1) + " & "
      + coefstars(ngmodel,1) + " & "
      + coefstars(nhnmodel,1) + " & "
      + coefstars(nnmodel,1) + " & "
      + coefstars(ntnmodel,1)
      + " \\\\")
print(" & "
      + se(nexpmodel,1) + " & "
      + se(ngmodel,1) + " & "
      + se(nhnmodel,1) + " & "
      + se(nnmodel,1) + " & "
      + se(ntnmodel,1)
      + " \\\\")
print("$\\beta_2$ & "
      + coefstars(nexpmodel,2) + " & "
      + coefstars(ngmodel,2) + " & "
      + coefstars(nhnmodel,2) + " & "
      + coefstars(nnmodel,2) + " & "
      + coefstars(ntnmodel,2)
      + " \\\\")
print(" & "
      + se(nexpmodel,2) + " & "
      + se(ngmodel,2) + " & "
      + se(nhnmodel,2) + " & "
      + se(nnmodel,2) + " & "
      + se(ntnmodel,2)
      + " \\\\")
print("$\\beta_3$ & "
      + coefstars(nexpmodel,3) + " & "
      + coefstars(ngmodel,3) + " & "
      + coefstars(nhnmodel,3) + " & "
      + coefstars(nnmodel,3) + " & "
      + coefstars(ntnmodel,3)
      + " \\\\")
print(" & "
      + se(nexpmodel,3) + " & "
      + se(ngmodel,3) + " & "
      + se(nhnmodel,3) + " & "
      + se(nnmodel,3) + " & "
      + se(ntnmodel,3)
      + " \\\\")
print("$\\beta_4$ & "
      + coefstars(nexpmodel,4) + " & "
      + coefstars(ngmodel,4) + " & "
      + coefstars(nhnmodel,4) + " & "
      + coefstars(nnmodel,4) + " & "
      + coefstars(ntnmodel,4)
      + " \\\\")
print(" & "
      + se(nexpmodel,4) + " & "
      + se(ngmodel,4) + " & "
      + se(nhnmodel,4) + " & "
      + se(nnmodel,4) + " & "
      + se(ntnmodel,4)
      + " \\\\")
print("\\midrule")
print("$\\ln\\sigma_V$ & "
      + coefstars(nexpmodel,5) + " & "
      + coefstars(ngmodel,5) + " & "
      + coefstars(nhnmodel,5) + " & "
      + coefstars(nnmodel,5) + " & "
      + coefstars(ntnmodel,5)
      + " \\\\")
print(" & "
      + se(nexpmodel,5) + " & "
      + se(ngmodel,5) + " & "
      + se(nhnmodel,5) + " & "
      + se(nnmodel,5) + " & "
      + se(ntnmodel,5)
      + " \\\\")
print("$\\ln\\sigma_U$ & "
      + coefstars(nexpmodel,6) + " & "
      + coefstars(ngmodel,6) + " & "
      + coefstars(nhnmodel,6) + " & "
      + coefstars(nnmodel,6) + " & "
      + coefstars(ntnmodel,6)
      + " \\\\")
print(" & "
      + se(nexpmodel,6) + " & "
      + se(ngmodel,6) + " & "
      + se(nhnmodel,6) + " & "
      + se(nnmodel,6) + " & "
      + se(ntnmodel,6)
      + " \\\\")
print("$\\ln\\mu$ & "
      + "0.00000" + " & "
      + coefstars(ngmodel,7) + " & "
      + str(round(np.log(0.5),5)) + " & "
      + coefstars(nnmodel,7) + " & "
      + "\multicolumn{1}{c}{--}"
      + " \\\\")
print(" & "
      + "\multicolumn{1}{c}{--}" + " & "
      + se(ngmodel,7) + " & "
      + "\multicolumn{1}{c}{--}" + " & "
      + se(nnmodel,7) + " & "
      + "\multicolumn{1}{c}{--}"
      + " \\\\")
print("\\midrule")
print("$\\sigma_V$ & "
      + sigmav(nexpmodel) + " & "
      + sigmav(ngmodel) + " & "
      + sigmav(nhnmodel) + " & "
      + sigmav(nnmodel) + " & "
      + sigmav(ntnmodel)
      + " \\\\")
print(" & "
      + sigmavse(nexpmodel) + " & "
      + sigmavse(ngmodel) + " & "
      + sigmavse(nhnmodel) + " & "
      + sigmavse(nnmodel) + " & "
      + sigmavse(ntnmodel)
      + " \\\\")
print("$\\sigma_U$ & "
      + sigmau(nexpmodel) + " & "
      + sigmau(ngmodel) + " & "
      + sigmau(nhnmodel) + " & "
      + sigmau(nnmodel) + " & "
      + sigmau(ntnmodel)
      + " \\\\")
print(" & "
      + sigmause(nexpmodel) + " & "
      + sigmause(ngmodel) + " & "
      + sigmause(nhnmodel) + " & "
      + sigmause(nnmodel) + " & "
      + sigmause(ntnmodel)
      + " \\\\")
print("$\\mu$ & "
      + "1.0000" + " & "
      + mu(ngmodel) + " & "
      + "0.5000" + " & "
      + mu(nnmodel) + " & "
      + mu(ntnmodel)
      + " \\\\")
print(" & "
      + "\multicolumn{1}{c}{--}" + " & "
      + muse(ngmodel) + " & "
      + "\multicolumn{1}{c}{--}" + " & "
      + muse(nnmodel) + " & "
      + muse(ntnmodel)
      + " \\\\")
print("\\midrule")
print("$\\ln L$ & "
      + lnL(nexpmodel) + " & "
      + lnL(ngmodel) + " & "
      + lnL(nhnmodel) + " & "
      + lnL(nnmodel) + " & "
      + lnL(ntnmodel)
      + " \\\\")
print("\\bottomrule")
print("\\multicolumn{6}{l}{\\footnotesize Standard errors in parentheses}\\\\")
print("\\multicolumn{6}{l}{\\footnotesize \\sym{*} \\(p<0.10\\), \\sym{**} \\(p<0.05\\), \\sym{***} \\(p<0.01\\)}\\\\")
print("\\end{tabular}}")
print("\\label{tab:parameterestimattes1}")
print("\\end{table}")
print("")

print("\\begin{table}[htbp]\\centering")
print("\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}")
print("\\caption{Likelihood ratio tests -- Application 1}")
print("{\\renewcommand{\\arraystretch}{0.5}%")
print("\\begin{tabular}{cc*{1}{D{.}{.}{8}}}")
print("\\toprule")
print(" Alternative model & Null model & \\multicolumn{1}{c}{Likelihood ratio} \\\\")
print("\\midrule")
print("N-G & N-EXP & "
      + LR(ngmodel,nexpmodel)
      + " \\\\")
print("N-N & N-HN & "
      + LR(nnmodel,nhnmodel)
      + " \\\\")
print("N-TN & N-HN & "
      + LR(ntnmodel,nhnmodel)
      + " \\\\")
print("\\bottomrule")
print("\\multicolumn{3}{l}{\\footnotesize \\sym{*} \\(p<0.10\\), \\sym{**} \\(p<0.05\\), \\sym{***} \\(p<0.01\\)}\\\\")
print("\\end{tabular}}")
print("\\label{tab:likelihoodratios1}")
print("\\end{table}")
print("")

nexpmodel = nexpmodel_rice
ngmodel = ngmodel_rice
nhnmodel = nhnmodel_rice
nnmodel = nnmodel_rice
ntnmodel = ntnmodel_rice

print("\\begin{table}[htbp]\\centering")
print("\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}")
print("\\caption{Parameter estimates -- Application 2}")
print("{\\renewcommand{\\arraystretch}{0.5}%")
print("\\begin{tabular}{l*{5}{D{.}{.}{8}}}")
print("\\toprule")
print(" & \\multicolumn{1}{c}{N-EXP} & \\multicolumn{1}{c}{N-G} & \multicolumn{1}{c}{N-HN} & \multicolumn{1}{c}{N-N} & \multicolumn{1}{c}{N-TN} \\\\")
print("\\midrule")
print("$\\beta_0$ & "
      + coefstars(nexpmodel,0) + " & "
      + coefstars(ngmodel,0) + " & "
      + coefstars(nhnmodel,0) + " & "
      + coefstars(nnmodel,0) + " & "
      + coefstars(ntnmodel,0)
      + " \\\\")
print(" & "
      + se(nexpmodel,0) + " & "
      + se(ngmodel,0) + " & "
      + se(nhnmodel,0) + " & "
      + se(nnmodel,0) + " & "
      + se(ntnmodel,0)
      + " \\\\")
print("$\\beta_1$ & "
      + coefstars(nexpmodel,1) + " & "
      + coefstars(ngmodel,1) + " & "
      + coefstars(nhnmodel,1) + " & "
      + coefstars(nnmodel,1) + " & "
      + coefstars(ntnmodel,1)
      + " \\\\")
print(" & "
      + se(nexpmodel,1) + " & "
      + se(ngmodel,1) + " & "
      + se(nhnmodel,1) + " & "
      + se(nnmodel,1) + " & "
      + se(ntnmodel,1)
      + " \\\\")
print("$\\beta_2$ & "
      + coefstars(nexpmodel,2) + " & "
      + coefstars(ngmodel,2) + " & "
      + coefstars(nhnmodel,2) + " & "
      + coefstars(nnmodel,2) + " & "
      + coefstars(ntnmodel,2)
      + " \\\\")
print(" & "
      + se(nexpmodel,2) + " & "
      + se(ngmodel,2) + " & "
      + se(nhnmodel,2) + " & "
      + se(nnmodel,2) + " & "
      + se(ntnmodel,2)
      + " \\\\")
print("$\\beta_3$ & "
      + coefstars(nexpmodel,3) + " & "
      + coefstars(ngmodel,3) + " & "
      + coefstars(nhnmodel,3) + " & "
      + coefstars(nnmodel,3) + " & "
      + coefstars(ntnmodel,3)
      + " \\\\")
print(" & "
      + se(nexpmodel,3) + " & "
      + se(ngmodel,3) + " & "
      + se(nhnmodel,3) + " & "
      + se(nnmodel,3) + " & "
      + se(ntnmodel,3)
      + " \\\\")
print("$\\beta_{11}$ & "
      + coefstars(nexpmodel,4) + " & "
      + coefstars(ngmodel,4) + " & "
      + coefstars(nhnmodel,4) + " & "
      + coefstars(nnmodel,4) + " & "
      + coefstars(ntnmodel,4)
      + " \\\\")
print(" & "
      + se(nexpmodel,4) + " & "
      + se(ngmodel,4) + " & "
      + se(nhnmodel,4) + " & "
      + se(nnmodel,4) + " & "
      + se(ntnmodel,4)
      + " \\\\")
print("$\\beta_{12}$ & "
      + coefstars(nexpmodel,5) + " & "
      + coefstars(ngmodel,5) + " & "
      + coefstars(nhnmodel,5) + " & "
      + coefstars(nnmodel,5) + " & "
      + coefstars(ntnmodel,5)
      + " \\\\")
print(" & "
      + se(nexpmodel,5) + " & "
      + se(ngmodel,5) + " & "
      + se(nhnmodel,5) + " & "
      + se(nnmodel,5) + " & "
      + se(ntnmodel,5)
      + " \\\\")
print("$\\beta_{13}$ & "
      + coefstars(nexpmodel,6) + " & "
      + coefstars(ngmodel,6) + " & "
      + coefstars(nhnmodel,6) + " & "
      + coefstars(nnmodel,6) + " & "
      + coefstars(ntnmodel,6)
      + " \\\\")
print(" & "
      + se(nexpmodel,6) + " & "
      + se(ngmodel,6) + " & "
      + se(nhnmodel,6) + " & "
      + se(nnmodel,6) + " & "
      + se(ntnmodel,6)
      + " \\\\")
print("$\\beta_{22}$ & "
      + coefstars(nexpmodel,7) + " & "
      + coefstars(ngmodel,7) + " & "
      + coefstars(nhnmodel,7) + " & "
      + coefstars(nnmodel,7) + " & "
      + coefstars(ntnmodel,7)
      + " \\\\")
print(" & "
      + se(nexpmodel,7) + " & "
      + se(ngmodel,7) + " & "
      + se(nhnmodel,7) + " & "
      + se(nnmodel,7) + " & "
      + se(ntnmodel,7)
      + " \\\\")
print("$\\beta_{23}$ & "
      + coefstars(nexpmodel,8) + " & "
      + coefstars(ngmodel,8) + " & "
      + coefstars(nhnmodel,8) + " & "
      + coefstars(nnmodel,8) + " & "
      + coefstars(ntnmodel,8)
      + " \\\\")
print(" & "
      + se(nexpmodel,8) + " & "
      + se(ngmodel,8) + " & "
      + se(nhnmodel,8) + " & "
      + se(nnmodel,8) + " & "
      + se(ntnmodel,8)
      + " \\\\")
print("$\\beta_{33}$ & "
      + coefstars(nexpmodel,9) + " & "
      + coefstars(ngmodel,9) + " & "
      + coefstars(nhnmodel,9) + " & "
      + coefstars(nnmodel,9) + " & "
      + coefstars(ntnmodel,9)
      + " \\\\")
print(" & "
      + se(nexpmodel,9) + " & "
      + se(ngmodel,9) + " & "
      + se(nhnmodel,9) + " & "
      + se(nnmodel,9) + " & "
      + se(ntnmodel,9)
      + " \\\\")
print("$\\beta_t$ & "
      + coefstars(nexpmodel,10) + " & "
      + coefstars(ngmodel,10) + " & "
      + coefstars(nhnmodel,10) + " & "
      + coefstars(nnmodel,10) + " & "
      + coefstars(ntnmodel,10)
      + " \\\\")
print(" & "
      + se(nexpmodel,10) + " & "
      + se(ngmodel,10) + " & "
      + se(nhnmodel,10) + " & "
      + se(nnmodel,10) + " & "
      + se(ntnmodel,10)
      + " \\\\")
print("\\midrule")
print("$\\ln\\sigma_V$ & "
      + coefstars(nexpmodel,11) + " & "
      + coefstars(ngmodel,11) + " & "
      + coefstars(nhnmodel,11) + " & "
      + coefstars(nnmodel,11) + " & "
      + coefstars(ntnmodel,11)
      + " \\\\")
print(" & "
      + se(nexpmodel,11) + " & "
      + se(ngmodel,11) + " & "
      + se(nhnmodel,11) + " & "
      + se(nnmodel,11) + " & "
      + se(ntnmodel,11)
      + " \\\\")
print("$\\ln\\sigma_U$ & "
      + coefstars(nexpmodel,12) + " & "
      + coefstars(ngmodel,12) + " & "
      + coefstars(nhnmodel,12) + " & "
      + coefstars(nnmodel,12) + " & "
      + coefstars(ntnmodel,12)
      + " \\\\")
print(" & "
      + se(nexpmodel,12) + " & "
      + se(ngmodel,12) + " & "
      + se(nhnmodel,12) + " & "
      + se(nnmodel,12) + " & "
      + se(ntnmodel,12)
      + " \\\\")
print("$\\ln\\mu$ & "
      + "0.00000" + " & "
      + coefstars(ngmodel,13) + " & "
      + str(round(np.log(0.5),5)) + " & "
      + coefstars(nnmodel,13) + " & "
      + "\multicolumn{1}{c}{--}"
      + " \\\\")
print(" & "
      + "\multicolumn{1}{c}{--}" + " & "
      + se(ngmodel,13) + " & "
      + "\multicolumn{1}{c}{--}" + " & "
      + se(nnmodel,13) + " & "
      + "\multicolumn{1}{c}{--}"
      + " \\\\")
print("\\midrule")
print("$\\sigma_V$ & "
      + sigmav(nexpmodel) + " & "
      + sigmav(ngmodel) + " & "
      + sigmav(nhnmodel) + " & "
      + sigmav(nnmodel) + " & "
      + sigmav(ntnmodel)
      + " \\\\")
print(" & "
      + sigmavse(nexpmodel) + " & "
      + sigmavse(ngmodel) + " & "
      + sigmavse(nhnmodel) + " & "
      + sigmavse(nnmodel) + " & "
      + sigmavse(ntnmodel)
      + " \\\\")
print("$\\sigma_U$ & "
      + sigmau(nexpmodel) + " & "
      + sigmau(ngmodel) + " & "
      + sigmau(nhnmodel) + " & "
      + sigmau(nnmodel) + " & "
      + sigmau(ntnmodel)
      + " \\\\")
print(" & "
      + sigmause(nexpmodel) + " & "
      + sigmause(ngmodel) + " & "
      + sigmause(nhnmodel) + " & "
      + sigmause(nnmodel) + " & "
      + sigmause(ntnmodel)
      + " \\\\")
print("$\\mu$ & "
      + "1.0000" + " & "
      + mu(ngmodel) + " & "
      + "0.5000" + " & "
      + mu(nnmodel) + " & "
      + mu(ntnmodel)
      + " \\\\")
print(" & "
      + "\multicolumn{1}{c}{--}" + " & "
      + muse(ngmodel) + " & "
      + "\multicolumn{1}{c}{--}" + " & "
      + muse(nnmodel) + " & "
      + muse(ntnmodel)
      + " \\\\")
print("\\midrule")
print("$\\ln L$ & "
      + lnL(nexpmodel) + " & "
      + lnL(ngmodel) + " & "
      + lnL(nhnmodel) + " & "
      + lnL(nnmodel) + " & "
      + lnL(ntnmodel)
      + " \\\\")
print("\\bottomrule")
print("\\multicolumn{6}{l}{\\footnotesize Standard errors in parentheses}\\\\")
print("\\multicolumn{6}{l}{\\footnotesize \\sym{*} \\(p<0.10\\), \\sym{**} \\(p<0.05\\), \\sym{***} \\(p<0.01\\)}\\\\")
print("\\end{tabular}}")
print("\\label{tab:parameterestimattes2}")
print("\\end{table}")
print("")

print("\\begin{table}[htbp]\\centering")
print("\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}")
print("\\caption{Likelihood ratio tests -- Application 2}")
print("{\\renewcommand{\\arraystretch}{0.5}%")
print("\\begin{tabular}{cc*{1}{D{.}{.}{8}}}")
print("\\toprule")
print(" Alternative model & Null model & \\multicolumn{1}{c}{Likelihood ratio} \\\\")
print("\\midrule")
print("N-G & N-EXP & "
      + LR(ngmodel,nexpmodel)
      + " \\\\")
print("N-N & N-HN & "
      + LR(nnmodel,nhnmodel)
      + " \\\\")
print("N-TN & N-HN & "
      + LR(ntnmodel,nhnmodel)
      + " \\\\")
print("\\bottomrule")
print("\\multicolumn{3}{l}{\\footnotesize \\sym{*} \\(p<0.10\\), \\sym{**} \\(p<0.05\\), \\sym{***} \\(p<0.01\\)}\\\\")
print("\\end{tabular}}")
print("\\label{tab:likelihoodratios2}")
print("\\end{table}")

nexpmodel = nexpmodel_electricitygb
ngmodel = ngmodel_electricitygb
nhnmodel = nhnmodel_electricitygb
nnmodel = nnmodel_electricitygb
ntnmodel = ntnmodel_electricitygb

print("\\begin{table}[htbp]\\centering")
print("\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}")
print("\\caption{Parameter estimates -- Application 3}")
print("{\\renewcommand{\\arraystretch}{0.5}%")
print("\\begin{tabular}{l*{5}{D{.}{.}{8}}}")
print("\\toprule")
print(" & \\multicolumn{1}{c}{N-EXP} & \\multicolumn{1}{c}{N-G} & \multicolumn{1}{c}{N-HN} & \multicolumn{1}{c}{N-N} & \multicolumn{1}{c}{N-TN} \\\\")
print("\\midrule")
print("$\\beta_0$ & "
      + coefstars(nexpmodel,0) + " & "
      + coefstars(ngmodel,0) + " & "
      + coefstars(nhnmodel,0) + " & "
      + coefstars(nnmodel,0) + " & "
      + coefstars(ntnmodel,0)
      + " \\\\")
print(" & "
      + se(nexpmodel,0) + " & "
      + se(ngmodel,0) + " & "
      + se(nhnmodel,0) + " & "
      + se(nnmodel,0) + " & "
      + se(ntnmodel,0)
      + " \\\\")
print("$\\beta_q$ & "
      + coefstars(nexpmodel,1) + " & "
      + coefstars(ngmodel,1) + " & "
      + coefstars(nhnmodel,1) + " & "
      + coefstars(nnmodel,1) + " & "
      + coefstars(ntnmodel,1)
      + " \\\\")
print(" & "
      + se(nexpmodel,1) + " & "
      + se(ngmodel,1) + " & "
      + se(nhnmodel,1) + " & "
      + se(nnmodel,1) + " & "
      + se(ntnmodel,1)
      + " \\\\")
print("$\\beta_{qq}$ & "
      + coefstars(nexpmodel,2) + " & "
      + coefstars(ngmodel,2) + " & "
      + coefstars(nhnmodel,2) + " & "
      + coefstars(nnmodel,2) + " & "
      + coefstars(ntnmodel,2)
      + " \\\\")
print(" & "
      + se(nexpmodel,2) + " & "
      + se(ngmodel,2) + " & "
      + se(nhnmodel,2) + " & "
      + se(nnmodel,2) + " & "
      + se(ntnmodel,2)
      + " \\\\")
print("$\\beta_u$ & "
      + coefstars(nexpmodel,3) + " & "
      + coefstars(ngmodel,3) + " & "
      + coefstars(nhnmodel,3) + " & "
      + coefstars(nnmodel,3) + " & "
      + coefstars(ntnmodel,3)
      + " \\\\")
print(" & "
      + se(nexpmodel,3) + " & "
      + se(ngmodel,3) + " & "
      + se(nhnmodel,3) + " & "
      + se(nnmodel,3) + " & "
      + se(ntnmodel,3)
      + " \\\\")
print("$\\beta_{uu}$ & "
      + coefstars(nexpmodel,4) + " & "
      + coefstars(ngmodel,4) + " & "
      + coefstars(nhnmodel,4) + " & "
      + coefstars(nnmodel,4) + " & "
      + coefstars(ntnmodel,4)
      + " \\\\")
print(" & "
      + se(nexpmodel,4) + " & "
      + se(ngmodel,4) + " & "
      + se(nhnmodel,4) + " & "
      + se(nnmodel,4) + " & "
      + se(ntnmodel,4)
      + " \\\\")
print("$\\beta_o$ & "
      + coefstars(nexpmodel,5) + " & "
      + coefstars(ngmodel,5) + " & "
      + coefstars(nhnmodel,5) + " & "
      + coefstars(nnmodel,5) + " & "
      + coefstars(ntnmodel,5)
      + " \\\\")
print(" & "
      + se(nexpmodel,5) + " & "
      + se(ngmodel,5) + " & "
      + se(nhnmodel,5) + " & "
      + se(nnmodel,5) + " & "
      + se(ntnmodel,5)
      + " \\\\")
print("$\\beta_{qu}$ & "
      + coefstars(nexpmodel,6) + " & "
      + coefstars(ngmodel,6) + " & "
      + coefstars(nhnmodel,6) + " & "
      + coefstars(nnmodel,6) + " & "
      + coefstars(ntnmodel,6)
      + " \\\\")
print(" & "
      + se(nexpmodel,6) + " & "
      + se(ngmodel,6) + " & "
      + se(nhnmodel,6) + " & "
      + se(nnmodel,6) + " & "
      + se(ntnmodel,6)
      + " \\\\")
print("$\\beta_{uo}$ & "
      + coefstars(nexpmodel,7) + " & "
      + coefstars(ngmodel,7) + " & "
      + coefstars(nhnmodel,7) + " & "
      + coefstars(nnmodel,7) + " & "
      + coefstars(ntnmodel,7)
      + " \\\\")
print(" & "
      + se(nexpmodel,7) + " & "
      + se(ngmodel,7) + " & "
      + se(nhnmodel,7) + " & "
      + se(nnmodel,7) + " & "
      + se(ntnmodel,7)
      + " \\\\")
print("$\\beta_{qo}$ & "
      + coefstars(nexpmodel,8) + " & "
      + coefstars(ngmodel,8) + " & "
      + coefstars(nhnmodel,8) + " & "
      + coefstars(nnmodel,8) + " & "
      + coefstars(ntnmodel,8)
      + " \\\\")
print(" & "
      + se(nexpmodel,8) + " & "
      + se(ngmodel,8) + " & "
      + se(nhnmodel,8) + " & "
      + se(nnmodel,8) + " & "
      + se(ntnmodel,8)
      + " \\\\")
print("$\\beta_{c}$ & "
      + coefstars(nexpmodel,9) + " & "
      + coefstars(ngmodel,9) + " & "
      + coefstars(nhnmodel,9) + " & "
      + coefstars(nnmodel,9) + " & "
      + coefstars(ntnmodel,9)
      + " \\\\")
print(" & "
      + se(nexpmodel,9) + " & "
      + se(ngmodel,9) + " & "
      + se(nhnmodel,9) + " & "
      + se(nnmodel,9) + " & "
      + se(ntnmodel,9)
      + " \\\\")
print("$\\beta_{s}$ & "
      + coefstars(nexpmodel,10) + " & "
      + coefstars(ngmodel,10) + " & "
      + coefstars(nhnmodel,10) + " & "
      + coefstars(nnmodel,10) + " & "
      + coefstars(ntnmodel,10)
      + " \\\\")
print(" & "
      + se(nexpmodel,10) + " & "
      + se(ngmodel,10) + " & "
      + se(nhnmodel,10) + " & "
      + se(nnmodel,10) + " & "
      + se(ntnmodel,10)
      + " \\\\")
print("$\\beta_{w}$ & "
      + coefstars(nexpmodel,11) + " & "
      + coefstars(ngmodel,11) + " & "
      + coefstars(nhnmodel,11) + " & "
      + coefstars(nnmodel,11) + " & "
      + coefstars(ntnmodel,11)
      + " \\\\")
print(" & "
      + se(nexpmodel,11) + " & "
      + se(ngmodel,11) + " & "
      + se(nhnmodel,11) + " & "
      + se(nnmodel,11) + " & "
      + se(ntnmodel,11)
      + " \\\\")
print("$\\beta_{cc}$ & "
      + coefstars(nexpmodel,12) + " & "
      + coefstars(ngmodel,12) + " & "
      + coefstars(nhnmodel,12) + " & "
      + coefstars(nnmodel,12) + " & "
      + coefstars(ntnmodel,12)
      + " \\\\")
print(" & "
      + se(nexpmodel,12) + " & "
      + se(ngmodel,12) + " & "
      + se(nhnmodel,12) + " & "
      + se(nnmodel,12) + " & "
      + se(ntnmodel,12)
      + " \\\\")
print("$\\beta_{ss}$ & "
      + coefstars(nexpmodel,13) + " & "
      + coefstars(ngmodel,13) + " & "
      + coefstars(nhnmodel,13) + " & "
      + coefstars(nnmodel,13) + " & "
      + coefstars(ntnmodel,13)
      + " \\\\")
print(" & "
      + se(nexpmodel,13) + " & "
      + se(ngmodel,13) + " & "
      + se(nhnmodel,13) + " & "
      + se(nnmodel,13) + " & "
      + se(ntnmodel,13)
      + " \\\\")
print("$\\beta_{cs}$ & "
      + coefstars(nexpmodel,14) + " & "
      + coefstars(ngmodel,14) + " & "
      + coefstars(nhnmodel,14) + " & "
      + coefstars(nnmodel,14) + " & "
      + coefstars(ntnmodel,14)
      + " \\\\")
print(" & "
      + se(nexpmodel,14) + " & "
      + se(ngmodel,14) + " & "
      + se(nhnmodel,14) + " & "
      + se(nnmodel,14) + " & "
      + se(ntnmodel,14)
      + " \\\\")
print("$\\beta_{cw}$ & "
      + coefstars(nexpmodel,15) + " & "
      + coefstars(ngmodel,15) + " & "
      + coefstars(nhnmodel,15) + " & "
      + coefstars(nnmodel,15) + " & "
      + coefstars(ntnmodel,15)
      + " \\\\")
print(" & "
      + se(nexpmodel,15) + " & "
      + se(ngmodel,15) + " & "
      + se(nhnmodel,15) + " & "
      + se(nnmodel,15) + " & "
      + se(ntnmodel,15)
      + " \\\\")
print("$\\beta_{sw}$ & "
      + coefstars(nexpmodel,16) + " & "
      + coefstars(ngmodel,16) + " & "
      + coefstars(nhnmodel,16) + " & "
      + coefstars(nnmodel,16) + " & "
      + coefstars(ntnmodel,16)
      + " \\\\")
print(" & "
      + se(nexpmodel,16) + " & "
      + se(ngmodel,16) + " & "
      + se(nhnmodel,16) + " & "
      + se(nnmodel,16) + " & "
      + se(ntnmodel,16)
      + " \\\\")
print("$\\beta_{cq}$ & "
      + coefstars(nexpmodel,17) + " & "
      + coefstars(ngmodel,17) + " & "
      + coefstars(nhnmodel,17) + " & "
      + coefstars(nnmodel,17) + " & "
      + coefstars(ntnmodel,17)
      + " \\\\")
print(" & "
      + se(nexpmodel,17) + " & "
      + se(ngmodel,17) + " & "
      + se(nhnmodel,17) + " & "
      + se(nnmodel,17) + " & "
      + se(ntnmodel,17)
      + " \\\\")
print("$\\beta_{sq}$ & "
      + coefstars(nexpmodel,18) + " & "
      + coefstars(ngmodel,18) + " & "
      + coefstars(nhnmodel,18) + " & "
      + coefstars(nnmodel,18) + " & "
      + coefstars(ntnmodel,18)
      + " \\\\")
print(" & "
      + se(nexpmodel,18) + " & "
      + se(ngmodel,18) + " & "
      + se(nhnmodel,18) + " & "
      + se(nnmodel,18) + " & "
      + se(ntnmodel,18)
      + " \\\\")
print("$\\beta_{wq}$ & "
      + coefstars(nexpmodel,19) + " & "
      + coefstars(ngmodel,19) + " & "
      + coefstars(nhnmodel,19) + " & "
      + coefstars(nnmodel,19) + " & "
      + coefstars(ntnmodel,19)
      + " \\\\")
print(" & "
      + se(nexpmodel,19) + " & "
      + se(ngmodel,19) + " & "
      + se(nhnmodel,19) + " & "
      + se(nnmodel,19) + " & "
      + se(ntnmodel,19)
      + " \\\\")
print("$\\beta_{cu}$ & "
      + coefstars(nexpmodel,20) + " & "
      + coefstars(ngmodel,20) + " & "
      + coefstars(nhnmodel,20) + " & "
      + coefstars(nnmodel,20) + " & "
      + coefstars(ntnmodel,20)
      + " \\\\")
print(" & "
      + se(nexpmodel,20) + " & "
      + se(ngmodel,20) + " & "
      + se(nhnmodel,20) + " & "
      + se(nnmodel,20) + " & "
      + se(ntnmodel,20)
      + " \\\\")
print("$\\beta_{su}$ & "
      + coefstars(nexpmodel,21) + " & "
      + coefstars(ngmodel,21) + " & "
      + coefstars(nhnmodel,21) + " & "
      + coefstars(nnmodel,21) + " & "
      + coefstars(ntnmodel,21)
      + " \\\\")
print(" & "
      + se(nexpmodel,21) + " & "
      + se(ngmodel,21) + " & "
      + se(nhnmodel,21) + " & "
      + se(nnmodel,21) + " & "
      + se(ntnmodel,21)
      + " \\\\")
print("$\\beta_{wu}$ & "
      + coefstars(nexpmodel,22) + " & "
      + coefstars(ngmodel,22) + " & "
      + coefstars(nhnmodel,22) + " & "
      + coefstars(nnmodel,22) + " & "
      + coefstars(ntnmodel,22)
      + " \\\\")
print(" & "
      + se(nexpmodel,22) + " & "
      + se(ngmodel,22) + " & "
      + se(nhnmodel,22) + " & "
      + se(nnmodel,22) + " & "
      + se(ntnmodel,22)
      + " \\\\")
print("\\bottomrule")
print("\\multicolumn{6}{l}{\\footnotesize Standard errors in parentheses}\\\\")
print("\\multicolumn{6}{l}{\\footnotesize \\sym{*} \\(p<0.10\\), \\sym{**} \\(p<0.05\\), \\sym{***} \\(p<0.01\\)}\\\\")
print("\\end{tabular}}")
print("\\label{tab:parameterestimattes3a}")
print("\\end{table}")
print("")
print("\\begin{table}[htbp]\\centering")
print("\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}")
print("\\caption{Parameter estimates -- Application 3 (continued)}")
print("{\\renewcommand{\\arraystretch}{0.5}%")
print("\\begin{tabular}{l*{5}{D{.}{.}{8}}}")
print("\\toprule")
print(" & \\multicolumn{1}{c}{N-EXP} & \\multicolumn{1}{c}{N-G} & \multicolumn{1}{c}{N-HN} & \multicolumn{1}{c}{N-N} & \multicolumn{1}{c}{N-TN} \\\\")
print("\\midrule")
print("$\\beta_{co}$ & "
      + coefstars(nexpmodel,23) + " & "
      + coefstars(ngmodel,23) + " & "
      + coefstars(nhnmodel,23) + " & "
      + coefstars(nnmodel,23) + " & "
      + coefstars(ntnmodel,23)
      + " \\\\")
print(" & "
      + se(nexpmodel,23) + " & "
      + se(ngmodel,23) + " & "
      + se(nhnmodel,23) + " & "
      + se(nnmodel,23) + " & "
      + se(ntnmodel,23)
      + " \\\\")
print("$\\beta_{so}$ & "
      + coefstars(nexpmodel,24) + " & "
      + coefstars(ngmodel,24) + " & "
      + coefstars(nhnmodel,24) + " & "
      + coefstars(nnmodel,24) + " & "
      + coefstars(ntnmodel,24)
      + " \\\\")
print(" & "
      + se(nexpmodel,24) + " & "
      + se(ngmodel,24) + " & "
      + se(nhnmodel,24) + " & "
      + se(nnmodel,24) + " & "
      + se(ntnmodel,24)
      + " \\\\")
print("$\\beta_{wo}$ & "
      + coefstars(nexpmodel,25) + " & "
      + coefstars(ngmodel,25) + " & "
      + coefstars(nhnmodel,25) + " & "
      + coefstars(nnmodel,25) + " & "
      + coefstars(ntnmodel,25)
      + " \\\\")
print(" & "
      + se(nexpmodel,25) + " & "
      + se(ngmodel,25) + " & "
      + se(nhnmodel,25) + " & "
      + se(nnmodel,25) + " & "
      + se(ntnmodel,25)
      + " \\\\")
print("$\\beta_{k}$ & "
      + coefstars(nexpmodel,26) + " & "
      + coefstars(ngmodel,26) + " & "
      + coefstars(nhnmodel,26) + " & "
      + coefstars(nnmodel,26) + " & "
      + coefstars(ntnmodel,26)
      + " \\\\")
print(" & "
      + se(nexpmodel,26) + " & "
      + se(ngmodel,26) + " & "
      + se(nhnmodel,26) + " & "
      + se(nnmodel,26) + " & "
      + se(ntnmodel,26)
      + " \\\\")
print("$\\beta_{kk}$ & "
      + coefstars(nexpmodel,27) + " & "
      + coefstars(ngmodel,27) + " & "
      + coefstars(nhnmodel,27) + " & "
      + coefstars(nnmodel,27) + " & "
      + coefstars(ntnmodel,27)
      + " \\\\")
print(" & "
      + se(nexpmodel,27) + " & "
      + se(ngmodel,27) + " & "
      + se(nhnmodel,27) + " & "
      + se(nnmodel,27) + " & "
      + se(ntnmodel,27)
      + " \\\\")
print("$\\beta_{kq}$ & "
      + coefstars(nexpmodel,28) + " & "
      + coefstars(ngmodel,28) + " & "
      + coefstars(nhnmodel,28) + " & "
      + coefstars(nnmodel,28) + " & "
      + coefstars(ntnmodel,28)
      + " \\\\")
print(" & "
      + se(nexpmodel,28) + " & "
      + se(ngmodel,28) + " & "
      + se(nhnmodel,28) + " & "
      + se(nnmodel,28) + " & "
      + se(ntnmodel,28)
      + " \\\\")
print("$\\beta_{kr}$ & "
      + coefstars(nexpmodel,29) + " & "
      + coefstars(ngmodel,29) + " & "
      + coefstars(nhnmodel,29) + " & "
      + coefstars(nnmodel,29) + " & "
      + coefstars(ntnmodel,29)
      + " \\\\")
print(" & "
      + se(nexpmodel,29) + " & "
      + se(ngmodel,29) + " & "
      + se(nhnmodel,29) + " & "
      + se(nnmodel,29) + " & "
      + se(ntnmodel,29)
      + " \\\\")
print("$\\beta_{ok}$ & "
      + coefstars(nexpmodel,30) + " & "
      + coefstars(ngmodel,30) + " & "
      + coefstars(nhnmodel,30) + " & "
      + coefstars(nnmodel,30) + " & "
      + coefstars(ntnmodel,30)
      + " \\\\")
print(" & "
      + se(nexpmodel,30) + " & "
      + se(ngmodel,30) + " & "
      + se(nhnmodel,30) + " & "
      + se(nnmodel,30) + " & "
      + se(ntnmodel,30)
      + " \\\\")
print("$\\beta_{ck}$ & "
      + coefstars(nexpmodel,31) + " & "
      + coefstars(ngmodel,31) + " & "
      + coefstars(nhnmodel,31) + " & "
      + coefstars(nnmodel,31) + " & "
      + coefstars(ntnmodel,31)
      + " \\\\")
print(" & "
      + se(nexpmodel,31) + " & "
      + se(ngmodel,31) + " & "
      + se(nhnmodel,31) + " & "
      + se(nnmodel,31) + " & "
      + se(ntnmodel,31)
      + " \\\\")
print("$\\beta_{sk}$ & "
      + coefstars(nexpmodel,32) + " & "
      + coefstars(ngmodel,32) + " & "
      + coefstars(nhnmodel,32) + " & "
      + coefstars(nnmodel,32) + " & "
      + coefstars(ntnmodel,32)
      + " \\\\")
print(" & "
      + se(nexpmodel,32) + " & "
      + se(ngmodel,32) + " & "
      + se(nhnmodel,32) + " & "
      + se(nnmodel,32) + " & "
      + se(ntnmodel,32)
      + " \\\\")
print("$\\beta_{wk}$ & "
      + coefstars(nexpmodel,33) + " & "
      + coefstars(ngmodel,33) + " & "
      + coefstars(nhnmodel,33) + " & "
      + coefstars(nnmodel,33) + " & "
      + coefstars(ntnmodel,33)
      + " \\\\")
print(" & "
      + se(nexpmodel,33) + " & "
      + se(ngmodel,33) + " & "
      + se(nhnmodel,33) + " & "
      + se(nnmodel,33) + " & "
      + se(ntnmodel,33)
      + " \\\\")
print("\\midrule")
print("$\\ln\\sigma_V$ & "
      + coefstars(nexpmodel,34) + " & "
      + coefstars(ngmodel,34) + " & "
      + coefstars(nhnmodel,34) + " & "
      + coefstars(nnmodel,34) + " & "
      + coefstars(ntnmodel,34)
      + " \\\\")
print(" & "
      + se(nexpmodel,34) + " & "
      + se(ngmodel,34) + " & "
      + se(nhnmodel,34) + " & "
      + se(nnmodel,34) + " & "
      + se(ntnmodel,34)
      + " \\\\")
print("$\\ln\\sigma_U$ & "
      + coefstars(nexpmodel,35) + " & "
      + coefstars(ngmodel,35) + " & "
      + coefstars(nhnmodel,35) + " & "
      + coefstars(nnmodel,35) + " & "
      + coefstars(ntnmodel,35)
      + " \\\\")
print(" & "
      + se(nexpmodel,35) + " & "
      + se(ngmodel,35) + " & "
      + se(nhnmodel,35) + " & "
      + se(nnmodel,35) + " & "
      + se(ntnmodel,35)
      + " \\\\")
print("$\\ln\\mu$ & "
      + "0.00000" + " & "
      + coefstars(ngmodel,36) + " & "
      + str(round(np.log(0.5),5)) + " & "
      + coefstars(nnmodel,36) + " & "
      + "\multicolumn{1}{c}{--}"
      + " \\\\")
print(" & "
      + "\multicolumn{1}{c}{--}" + " & "
      + se(ngmodel,36) + " & "
      + "\multicolumn{1}{c}{--}" + " & "
      + se(nnmodel,36) + " & "
      + "\multicolumn{1}{c}{--}"
      + " \\\\")
print("\\midrule")
print("$\\sigma_V$ & "
      + sigmav(nexpmodel) + " & "
      + sigmav(ngmodel) + " & "
      + sigmav(nhnmodel) + " & "
      + sigmav(nnmodel) + " & "
      + sigmav(ntnmodel)
      + " \\\\")
print(" & "
      + sigmavse(nexpmodel) + " & "
      + sigmavse(ngmodel) + " & "
      + sigmavse(nhnmodel) + " & "
      + sigmavse(nnmodel) + " & "
      + sigmavse(ntnmodel)
      + " \\\\")
print("$\\sigma_U$ & "
      + sigmau(nexpmodel) + " & "
      + sigmau(ngmodel) + " & "
      + sigmau(nhnmodel) + " & "
      + sigmau(nnmodel) + " & "
      + sigmau(ntnmodel)
      + " \\\\")
print(" & "
      + sigmause(nexpmodel) + " & "
      + sigmause(ngmodel) + " & "
      + sigmause(nhnmodel) + " & "
      + sigmause(nnmodel) + " & "
      + sigmause(ntnmodel)
      + " \\\\")
print("$\\mu$ & "
      + "1.0000" + " & "
      + mu(ngmodel) + " & "
      + "0.5000" + " & "
      + mu(nnmodel) + " & "
      + mu(ntnmodel)
      + " \\\\")
print(" & "
      + "\multicolumn{1}{c}{--}" + " & "
      + muse(ngmodel) + " & "
      + "\multicolumn{1}{c}{--}" + " & "
      + muse(nnmodel) + " & "
      + muse(ntnmodel)
      + " \\\\")
print("\\midrule")
print("$\\ln L$ & "
      + lnL(nexpmodel) + " & "
      + lnL(ngmodel) + " & "
      + lnL(nhnmodel) + " & "
      + lnL(nnmodel) + " & "
      + lnL(ntnmodel)
      + " \\\\")
print("\\bottomrule")
print("\\multicolumn{6}{l}{\\footnotesize Standard errors in parentheses}\\\\")
print("\\multicolumn{6}{l}{\\footnotesize \\sym{*} \\(p<0.10\\), \\sym{**} \\(p<0.05\\), \\sym{***} \\(p<0.01\\)}\\\\")
print("\\end{tabular}}")
print("\\label{tab:parameterestimattes3b}")
print("\\end{table}")
print("")

print("\\begin{table}[htbp]\\centering")
print("\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}")
print("\\caption{Model estimation times}")
print("{\\renewcommand{\\arraystretch}{0.5}%")
print("\\begin{tabular}{cc*{1}{D{.}{.}{8}}}")
print("\\toprule")
print(" Application & Model & \\multicolumn{1}{c}{Time (seconds)} \\\\")
print("\\midrule")
print("  & N-EXP & "
      + format(nexpmodel_electricity_time,'.4f')
      + " \\\\")
print("  & N-G & "
      + format(ngmodel_electricity_time,'.4f')
      + " \\\\")
print("1 & N-HN & "
      + format(nhnmodel_electricity_time,'.4f')
      + " \\\\")
print("  & N-N & "
      + format(nnmodel_electricity_time,'.4f')
      + " \\\\")
print("  & N-TN & "
      + format(ntnmodel_electricity_time,'.4f')
      + " \\\\")
print("\\midrule")
print(" & N-EXP & "
      + format(nexpmodel_rice_time,'.4f')
      + " \\\\")
print("  & N-G & "
      + format(ngmodel_rice_time,'.4f')
      + " \\\\")
print("2 & N-HN & "
      + format(nhnmodel_rice_time,'.4f')
      + " \\\\")
print("  & N-N & "
      + format(nnmodel_rice_time,'.4f')
      + " \\\\")
print("  & N-TN & "
      + format(ntnmodel_rice_time,'.4f')
      + " \\\\")
print("\\midrule")
print("  & N-EXP & "
      + format(nexpmodel_electricitygb_time,'.4f')
      + " \\\\")
print("  & N-G & "
      + format(ngmodel_electricitygb_time,'.4f')
      + " \\\\")
print("3 & N-HN & "
      + format(nhnmodel_electricitygb_time,'.4f')
      + " \\\\")
print("  & N-N & "
      + format(nnmodel_electricitygb_time,'.4f')
      + " \\\\")
print("  & N-TN & "
      + format(ntnmodel_electricitygb_time,'.4f')
      + " \\\\")
print("\\bottomrule")
print("\\end{tabular}}")
print("\\label{tab:estimationtime}")
print("\\end{table}")

print("\\begin{table}[htbp]\\centering")
print("\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}")
print("\\caption{Likelihood ratio tests}")
print("{\\renewcommand{\\arraystretch}{0.5}%")
print("\\begin{tabular}{ccc*{1}{D{.}{.}{8}}}")
print("\\toprule")
print(" Application & Alternative model & Null model & \\multicolumn{1}{c}{Likelihood ratio} \\\\")
print("\\midrule")
print(" & N-G & N-EXP & "
      + LR(ngmodel_electricity,nexpmodel_electricity)
      + " \\\\")
print("1 & N-N & N-HN & "
      + LR(nnmodel_electricity,nhnmodel_electricity)
      + " \\\\")
print(" & N-TN & N-HN & "
      + LR(ntnmodel_electricity,nhnmodel_electricity)
      + " \\\\")
print("\\midrule")
print(" & N-G & N-EXP & "
      + LR(ngmodel_rice,nexpmodel_rice)
      + " \\\\")
print("2 & N-N & N-HN & "
      + LR(nnmodel_rice,nhnmodel_rice)
      + " \\\\")
print(" & N-TN & N-HN & "
      + LR(ntnmodel_rice,nhnmodel_rice)
      + " \\\\")
print("\\midrule")
print(" & N-G & N-EXP & "
      + LR(ngmodel_electricitygb,nexpmodel_electricitygb)
      + " \\\\")
print("3 & N-N & N-HN & "
      + LR(nnmodel_electricitygb,nhnmodel_electricitygb)
      + " \\\\")
print(" & N-TN & N-HN & "
      + LR(ntnmodel_electricitygb,nhnmodel_electricitygb)
      + " \\\\")
print("\\bottomrule")
print("\\multicolumn{3}{l}{\\footnotesize \\sym{*} \\(p<0.10\\), \\sym{**} \\(p<0.05\\), \\sym{***} \\(p<0.01\\)}\\\\")
print("\\end{tabular}}")
print("\\label{tab:likelihoodratios2}")
print("\\end{table}")
