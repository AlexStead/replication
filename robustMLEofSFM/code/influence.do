ssc install colorscatter, replace
ssc install sfcross, replace
ssc install estout, replace
net install github, from("https://haghish.github.io/github/")
github install AlexStead/rfrontier, version(1.1.0)
clear all
set maxvar 100000
import delimited "https://github.com/AlexStead/data/tree/main/replication/robustMLEofSFM/ElectricityCosts.csv"
keep cost output lprice cprice fprice
summarize fprice
gen double lnf = ln(fprice/r(mean))
gen double lnc = ln(cost) - lnf
summarize lprice
gen double lnw = ln(lprice/r(mean)) - lnf
summarize cprice
gen double lnr = ln(cprice/r(mean)) - lnf
summarize output
gen double lnq = ln(output/r(mean))
gen double lnq2 = lnq^2
local yvar lnc
local xvars lnq lnq2 lnw lnr
graph drop _all

eststo nhn: frontier `yvar' `xvars', cost
nlcom (sigma_v: exp(_b[lnsig2v:_cons]/2)) (sigma_u: exp(_b[lnsig2u:_cons]/2))
estadd local blank0 "`:di ""'"
estadd local b_sigma_v "`:di %5.3f r(b)[1,1]'"
estadd local se_sigma_v "(`:di %5.3f sqrt(r(V)[1,1])')"
estadd local blank1 "`:di ""'"
estadd local b_sigma_u "`:di %5.3f r(b)[1,2]'"
estadd local se_sigma_u "(`:di %5.3f sqrt(r(V)[2,2])')"
estadd local b_df "\infty"
estadd local se_df "-"
local npar = colsof(e(b))
local s = -1
local sigmav sqrt(exp(xb(#2)))
local sigmau sqrt(exp(xb(#3)))
local sigma sqrt((`sigmav')^2+(`sigmau')^2)
local lambda (`sigmau')/(`sigmav')
local gamma (`sigmau')^2/(`sigma')^2
local epsilon `yvar'-xb(#1)
local mustar -`s'*(`epsilon')*((`sigmau')/(`sigma'))^2
local sigmastar ((`sigmau')*(`sigmav')/(`sigma'))
local invmills normalden(-(`epsilon')*(`lambda')/(`sigma'))/normal(-(`epsilon')*(`lambda')/(`sigma'))
predictnl double logl = -1/2*ln(2)+1/2*ln((`sigma')^2)+1/2*ln(_pi)+1/2*((`epsilon')/(`sigma'))^2-ln(normal(-`s'*(`epsilon')*(`lambda')/(`sigma'))), g(stub) force
mkmat stub*, matrix(score)
matrix information = e(V)
matrix influence = -(information*score')'
svmat double influence, name(inf_nhn)
replace inf_nhn`=`npar'-1'= 1/2*inf_nhn`=`npar'-1'*sqrt(exp(_b[lnsig2v:_cons]))
replace inf_nhn`=`npar''= 1/2*inf_nhn`=`npar''*sqrt(exp(_b[lnsig2u:_cons]))
gen double leverage_nhn = inf_nhn1*lnq + inf_nhn2*lnq2 + inf_nhn3*lnw + inf_nhn4*lnr + inf_nhn5
gen double abs_lev_nhn = abs(leverage_nhn)
predict double xb
gen double resid_nhn = `yvar'-xb
label variable resid_nhn "Residual"
label variable inf_nhn`npar' "Influence"
label variable lnq "lnq"
predictnl double eff_nhn = (1-normal(`sigmastar'-`mustar'/`sigmastar'))/(1-normal(-`mustar'/`sigmastar'))*exp(-`mustar'+1/2*`sigmastar'^2), g(gradient_nhn) force
mkmat gradient_nhn*, matrix(eff_gradient)
matrix inf_eff_nhn_ob = eff_gradient*influence'
matrix inf_eff_nhn_eff = inf_eff_nhn_ob'
svmat double inf_eff_nhn_ob, name(inf_eff_nhn_ob)
svmat double inf_eff_nhn_eff, name(inf_eff_nhn_eff)

colorscatter inf_nhn7 resid_nhn abs_lev_nhn, scatter_options(xtitle("{it:`=ustrunescape("\u03B5\u0302")'{subscript:𝑖}}", height(20)) ytitle("Influence")) scheme(s1color) rgb_low("0 255 255 0") rgb_high("0 0 255 0") symbol_opacity(50)
colorscatter inf_nhn1 resid_nhn abs_lev_nhn, scatter_options(xtitle("{it:`=ustrunescape("\u03B5\u0302")'𝑖}", height(20)) ytitle("{fontface "CMU Serif":Influence}") msymb(Oh)) scheme(s1color) rgb_low("255 255 255 255") rgb_high("50 50 50 50")

eststo nexp: frontier `yvar' `xvars', cost d(e)
predict double xb_nexp
gen double resid_nexp = `yvar'-xb_nexp
local epsilon `yvar'-xb(#1)
nlcom (sigma_v: exp(_b[lnsig2v:_cons]/2)) (sigma_u: exp(_b[lnsig2u:_cons]/2))
estadd local blank0 "`:di ""'"
estadd local b_sigma_v "`:di %5.3f r(b)[1,1]'"
estadd local se_sigma_v "(`:di %5.3f sqrt(r(V)[1,1])')"
estadd local blank1 "`:di ""'"
estadd local b_sigma_u "`:di %5.3f r(b)[1,2]'"
estadd local se_sigma_u "(`:di %5.3f sqrt(r(V)[2,2])')"
estadd local b_df "\infty"
estadd local se_df "-"
predictnl double logl_nexp = ln(`sigmau')-1/2*(1/(`lambda'))^2-`s'*(`epsilon')/(`sigmau')-ln(normal(-`s'*(`epsilon')/(`sigmav')-1/(`lambda'))), g(stub_nexp) force
mkmat stub_nexp*, matrix(score)
matrix information = e(V)
matrix influence = -(information*score')'
svmat double influence, name(inf_nexp)
replace inf_nexp`=`npar'-1'= 1/2*inf_nexp`=`npar'-1'*sqrt(exp(_b[lnsig2v:_cons]))
replace inf_nexp`=`npar''= 1/2*inf_nexp`=`npar''*sqrt(exp(_b[lnsig2u:_cons]))
gen double leverage_nexp = inf_nexp1*lnq + inf_nexp2*lnq2 + inf_nexp3*lnw + inf_nexp4*lnr + inf_nexp5
gen abs_lev_nexp = abs(leverage_nexp)
predictnl double eff_nexp = (1-normal(`s'*(`epsilon')/`sigmav'+`sigmav'/`sigmau'+`sigmav'))/(1-normal(`s'*(`epsilon')/`sigmav'+`sigmav'/`sigmau'))*exp(`s'*(`epsilon')+(`sigmav')^2/`sigmau'+((`sigmav')^2)/2), g(gradient_nexp) force
mkmat gradient_nexp*, matrix(eff_gradient)
matrix inf_eff_nexp_ob = eff_gradient*influence'
matrix inf_eff_nexp_eff = inf_eff_nexp_ob'
svmat double inf_eff_nexp_ob, name(inf_eff_nexp_ob)
svmat double inf_eff_nexp_eff, name(inf_eff_nexp_eff)

eststo cauchyhn: rfrontier `yvar' `xvars', cost v(c)
nlcom (sigma_v: exp(_b[lnsigma_v:_cons])) (sigma_u: exp(_b[lnsigma_u:_cons]))
estadd local blank0 "`:di ""'"
estadd local b_sigma_v "`:di %5.3f r(b)[1,1]'"
estadd local se_sigma_v "(`:di %5.3f sqrt(r(V)[1,1])')"
estadd local blank1 "`:di ""'"
estadd local b_sigma_u "`:di %5.3f r(b)[1,2]'"
estadd local se_sigma_u "(`:di %5.3f sqrt(r(V)[2,2])')"
estadd local blank2 "`:di ""'"
estadd local b_df "1.000"
estadd local se_df "-"
predict inf_chn, influence
gen double leverage_chn = inf_chn1*lnq + inf_chn2*lnq2 + inf_chn3*lnw + inf_chn4*lnr + inf_chn5
gen double abs_lev_chn = abs(leverage_chn)
predict double xb_chn
gen double resid_chn = `yvar'-xb_chn
predict double eff_chn, bc
predict double inf_eff_chn_ob, bcinf

eststo studhn: rfrontier `yvar' `xvars', cost
nlcom (sigma_v: exp(_b[lnsigma_v:_cons])) (sigma_u: exp(_b[lnsigma_u:_cons])) (sigma_u: exp(_b[lndf:_cons]))
estadd local blank0 "`:di ""'"
estadd local b_sigma_v "`:di %5.3f r(b)[1,1]'"
estadd local se_sigma_v "(`:di %5.3f sqrt(r(V)[1,1])')"
estadd local blank1 "`:di ""'"
estadd local b_sigma_u "`:di %5.3f r(b)[1,2]'"
estadd local se_sigma_u "(`:di %5.3f sqrt(r(V)[2,2])')"
estadd local blank2 "`:di ""'"
estadd local b_df "`:di %5.3f r(b)[1,3]'"
estadd local se_df "(`:di %5.3f sqrt(r(V)[3,3])')"
predict double inf_studhn, influence
gen double leverage_studhn = inf_studhn1*lnq + inf_studhn2*lnq2 + inf_studhn3*lnw + inf_studhn4*lnr + inf_studhn5
gen double abs_lev_studhn = abs(leverage_studhn)
predict double xb_studhn
gen double resid_studhn = `yvar'-xb_studhn
predict double eff_studhn, bc
rename inf_studhn7 ph
rename inf_studhn8 inf_studhn7
rename ph inf_studhn8
scatter inf_studhn7 resid_studhn
predict double inf_eff_studhn_ob, bcinf

eststo cauchyexp: rfrontier `yvar' `xvars', cost v(c) u(e)
nlcom (sigma_v: exp(_b[lnsigma_v:_cons])) (sigma_u: exp(_b[lnsigma_u:_cons]))
estadd local blank0 "`:di ""'"
estadd local b_sigma_v "`:di %5.3f r(b)[1,1]'"
estadd local se_sigma_v "(`:di %5.3f sqrt(r(V)[1,1])')"
estadd local blank1 "`:di ""'"
estadd local b_sigma_u "`:di %5.3f r(b)[1,2]'"
estadd local se_sigma_u "(`:di %5.3f sqrt(r(V)[2,2])')"
estadd local blank2 "`:di ""'"
estadd local b_df "1.000"
estadd local se_df "-"
predict double inf_cexp, influence
gen double leverage_cexp = inf_cexp1*lnq + inf_cexp2*lnq2 + inf_cexp3*lnw + inf_cexp4*lnr + inf_cexp5
gen double abs_lev_cexp = abs(leverage_cexp)
predict double xb_cexp
gen double resid_cexp = lnc-xb_cexp
predict double eff_cexp, bc
predict double inf_eff_cexp_ob, bcinf

eststo studexp: rfrontier `yvar' `xvars', cost u(e)
nlcom (sigma_v: exp(_b[lnsigma_v:_cons])) (sigma_u: exp(_b[lnsigma_u:_cons])) (sigma_u: exp(_b[lndf:_cons]))
estadd local blank0 "`:di ""'"
estadd local b_sigma_v "`:di %5.3f r(b)[1,1]'"
estadd local se_sigma_v "(`:di %5.3f sqrt(r(V)[1,1])')"
estadd local blank1 "`:di ""'"
estadd local b_sigma_u "`:di %5.3f r(b)[1,2]'"
estadd local se_sigma_u "(`:di %5.3f sqrt(r(V)[2,2])')"
estadd local blank2 "`:di ""'"
estadd local b_df "`:di %5.3f r(b)[1,3]'"
estadd local se_df "(`:di %5.3f sqrt(r(V)[3,3])')"
predict double inf_studexp, influence
gen double leverage_studexp = inf_studexp1*lnq + inf_studexp2*lnq2 + inf_studexp3*lnw + inf_studexp4*lnr + inf_studexp5
gen double abs_lev_studexp = abs(leverage_studexp)
predict double xb_studexp
gen double resid_studexp = lnc-xb_studexp
predict double eff_studexp, bc
rename inf_studexp7 ph
rename inf_studexp8 inf_studexp7
rename ph inf_studexp8
predict double inf_eff_studexp_ob, bcinf

graph set window fontface "Latin Modern Math"
graph set window fontfaceserif "CMU Serif"

graph set eps fontface "Latin Modern Math"
graph set eps fontfaceserif "CMU Serif"
graph set svg fontface "Latin Modern Math"
graph set svg fontfaceserif "CMU Serif"

foreach vdist in n c stud{
foreach udist in hn exp {
format inf_`vdist'`udist'1 %9.3f
format inf_`vdist'`udist'2 %9.3f
format inf_`vdist'`udist'3 %9.2f
format inf_`vdist'`udist'4 %9.2f
format inf_`vdist'`udist'5 %9.2f
format inf_`vdist'`udist'6 %9.2f
format inf_`vdist'`udist'7 %9.2f
}
}

local fullnamen "Normal"
local fullnamec "Cauchy"
local fullnamestud "Student's t"
local fullnamehn "half normal"
local fullnameexp "exponential"

forval i=1/7{
local model = 0
foreach udist in hn exp {
foreach vdist in n c stud {
format resid_`vdist'`udist' %9.1f
local model = `model' + 1
colorscatter inf_`vdist'`udist'`i' resid_`vdist'`udist' abs_lev_`vdist'`udist', scatter_options(title("{stSerif:Model (`model') (`fullname`vdist''-`fullname`udist'')}") xtitle("{it:`=ustrunescape("\u03B5\u0302")'{subscript:𝑖}}") ytitle("{stSerif:Influence}") msymb(Oh)) scheme(s1color) rgb_high("255 255 255 255") rgb_low("50 50 50 50") symbol_opacity(50) name(`vdist'`udist'`i') legend(off)
}
}
gr combine nhn`i' chn`i' studhn`i' nexp`i' cexp`i' studexp`i', ycommon xcommon name(graph`i') col(3) graphregion(margin(zero)) altshrink
graph export "C:\Users\traads\Documents\graph`i'.svg", as(svg) name("graph`i'") replace fontface("Latin Modern Math") fontfaceserif("CMU Serif")
graph drop nhn`i' chn`i' studhn`i' nexp`i' cexp`i' studexp`i' graph`i'
}

local fullnamen "Normal"
local fullnamec "Cauchy"
local fullnamestud "Student's t"
local fullnamehn "half normal"
local fullnameexp "exponential"
local model = 0
foreach udist in hn exp {
local model = `model' + 3
colorscatter inf_stud`udist'8 resid_stud`udist' abs_lev_stud`udist', scatter_options(title("{stSerif:Model `model' (`fullnamestud'-`fullname`udist'')}") xtitle("{it:`=ustrunescape("\u03B5\u0302")'{subscript:𝑖}}") ytitle("{stSerif:Influence}") msymb(Oh)) scheme(s1color) rgb_high("255 255 255 255") rgb_low("50 50 50 50") symbol_opacity(50) name(stud`udist'8) legend(off)
}
gr combine studhn8 studexp8 studhn8 studexp8 studhn8 studexp8, ycommon xcommon name(graph8) col(3) graphregion(margin(zero)) altshrink
graph export "C:\Users\traads\Documents\graph8.svg", as(svg) name("graph8") replace fontface("Latin Modern Math") fontfaceserif("CMU Serif")

label var lnq "\(\beta_1\:(\ln{q})\)"
label var lnq2 "\(\beta_2\:(\mathrm{ln}^2q)\)"
label var lnw "\(\beta_3\:(\ln{w})\)"
label var lnr "\(\beta_4\:(\ln{r})\)"
esttab nhn cauchyhn studhn nexp cauchyexp studexp  using "parameterestimates.tex", b(3) se(3) nomtitle label booktabs alignment(D{.}{.}{-1}) star(* 0.10 ** 0.05 *** 0.01) noobs  eqlabels("") scalar("blank0 hspace" "b_sigma_v sigv" "se_sigma_v hspace" "blank1 hspace" "b_sigma_u sigma_u" "se_sigma_u hspace" "blank2 hspace" "b_df df" "se_df hspace") drop(lnsig2v:_cons lnsig2u:_cons lnsigma_v:_cons lnsigma_u:_cons lndf:_cons) substitute(sigv \(\sigma_v\) sigma_u \(\sigma_u\) df \(a\) hspace \(\) Constant \(\beta_0\)) addnote("N.B. where \(a\to\infty, v_i\sim N(0,\sigma_v^2)\) and where \(a=1, v_i\sim\mathrm{Cauchy}(0,\sigma_v)\). In these models, \(a\) is fixed")  mgroups("\(u_i\sim N^+(0,\sigma_v^2)\)" "\(u_i\sim \mathrm{Exponential}(0,\sigma_v)\)", pattern(1 0 0 1 0 0) span prefix(\multicolumn{@span}{c}{) suffix(}) erepeat(\cmidrule(lr){@span})) title("Parameter estimates") replace


gen n = _n
mat eff_inf_range = J(123,1,.)
forval i=1/123{
egen double inf_eff_max_`i' = rowmax(inf_eff_nhn_ob`i' inf_eff_chn_ob`i' inf_eff_studhn_ob`i')
egen double inf_eff_min_`i' = rowmin(inf_eff_nhn_ob`i' inf_eff_chn_ob`i' inf_eff_studhn_ob`i')
summarize inf_eff_max_`i'
local max = r(max)
summarize inf_eff_min_`i'
local min = r(min)
mat eff_inf_range[`i',1] = `=`max'-`min''
drop inf_eff_max_`i' inf_eff_min_`i' 
}
svmat double eff_inf_range
egen rank = rank(-eff_inf_range)
sort rank

graph drop _all
format eff_nhn %9.1f
format eff_chn %9.1f
format eff_studhn %9.1f
forval i=1/6{
local j = n[`i']
format inf_eff_nhn_ob`j' %9.2f
format inf_eff_chn_ob`j' %9.2f
format inf_eff_studhn_ob`j' %9.2f
twoway (scatter inf_eff_nhn_ob`j' eff_nhn, col(black) msymbol(Oh)) (scatter inf_eff_chn_ob`j' eff_chn, col(black) msymbol(Sh)) (scatter inf_eff_studhn_ob`j' eff_studhn, col(black) msymbol(Th)), legend(off) scheme(s1color) name(scatter`i') ytitle("{stSerif:Influence}") xtitle("{stSerif:Predicted efficiency}", height(6))
}

twoway (scatter inf_eff_nhn_ob1 eff_nhn, col(black) msymbol(Oh) msize(14) mlwidth(3)) (scatter inf_eff_chn_ob1 eff_chn, col(black) msymbol(Sh) msize(14) mlwidth(3)) (scatter inf_eff_studhn_ob1 eff_studhn, col(black) msymbol(Th) msize(14) mlwidth(3)) (scatter inf_eff_nhn_ob1 eff_nhn, col(white) msymbol(O) msize(15) mlwidth(3)) (scatter inf_eff_chn_ob1 eff_chn, col(white) msymbol(S) msize(15) mlwidth(3)) (scatter inf_eff_studhn_ob1 eff_studhn, col(white) msymbol(T) msize(15) mlwidth(3)), scheme(s1color) legend(label(1 "Model 1") label(2 "Model 2") label(3 "Model 3") order(1 2 3) cols(3) position(12) ring(0) textw(20) size(30) colgap(120) region(color(white))) xsc(noline) ysc(noline) xti("") yti("") xla(none) yla(none) plotregion(style(none) margin(zero)) graphregion(style(none) margin(zero)) name(legend)  fysize(5)

graph combine scatter1 scatter2 scatter3 scatter4 scatter5 scatter6 legend, xcommon ycommon graphregion(margin(zero)) altshrink scheme(s1color) col(3) name(scatters1) hole(7 9)
graph export "C:\Users\traads\Documents\scatters1.svg", as(svg) name("scatters1") replace fontface("Latin Modern Math") fontfaceserif("CMU Serif")

sort n
drop rank n eff_inf_range

gen n = _n
mat eff_inf_range = J(123,1,.)
forval i=1/123{
egen double inf_eff_max_`i' = rowmax(inf_eff_nexp_ob`i' inf_eff_cexp_ob`i' inf_eff_studexp_ob`i')
egen double inf_eff_min_`i' = rowmin(inf_eff_nexp_ob`i' inf_eff_cexp_ob`i' inf_eff_studexp_ob`i')
summarize inf_eff_max_`i'
local max = r(max)
summarize inf_eff_min_`i'
local min = r(min)
mat eff_inf_range[`i',1] = `=`max'-`min''
drop inf_eff_max_`i' inf_eff_min_`i'
}
svmat double eff_inf_range
egen rank = rank(-eff_inf_range)
sort rank

graph drop _all
format eff_nexp %9.1f
format eff_cexp %9.1f
format eff_studexp %9.1f
forval i=1/6{
local j = n[`i']
format inf_eff_nexp_ob`j' %9.2f
format inf_eff_cexp_ob`j' %9.2f
format inf_eff_studexp_ob`j' %9.2f
twoway (scatter inf_eff_nexp_ob`j' eff_nexp, col(black) msymbol(Oh)) (scatter inf_eff_cexp_ob`j' eff_cexp, col(black) msymbol(Sh)) (scatter inf_eff_studexp_ob`j' eff_studexp, col(black) msymbol(Th)), legend(off) scheme(s1color) name(scatter`i') ytitle("{stSerif:Influence}") xtitle("{stSerif:Predicted efficiency}", height(6))
}

twoway (scatter inf_eff_nexp_ob1 eff_nexp, col(black) msymbol(Oh) msize(14) mlwidth(3)) (scatter inf_eff_cexp_ob1 eff_cexp, col(black) msymbol(Sh) msize(14) mlwidth(3)) (scatter inf_eff_studexp_ob1 eff_studexp, col(black) msymbol(Th) msize(14) mlwidth(3)) (scatter inf_eff_nexp_ob1 eff_nexp, col(white) msymbol(O) msize(15) mlwidth(3)) (scatter inf_eff_cexp_ob1 eff_cexp, col(white) msymbol(S) msize(15) mlwidth(3)) (scatter inf_eff_studexp_ob1 eff_studexp, col(white) msymbol(T) msize(15) mlwidth(3)), scheme(s1color) legend(label(1 "Model 4") label(2 "Model 5") label(3 "Model 6") order(1 2 3) cols(3) position(12) ring(0) textw(20) size(30) colgap(120) region(color(white))) xsc(noline) ysc(noline) xti("") yti("") xla(none) yla(none) plotregion(style(none) margin(zero)) graphregion(style(none) margin(zero)) name(legend)  fysize(5)

graph combine scatter1 scatter2 scatter3 scatter4 scatter5 scatter6 legend, xcommon ycommon graphregion(margin(zero)) altshrink scheme(s1color) col(3) name(scatters2) hole(7 9)
graph export "C:\Users\traads\Documents\scatters2.svg", as(svg) name("scatters2") replace fontface("Latin Modern Math") fontfaceserif("CMU Serif")

sort n
drop rank n eff_inf_range

frontier lnc lnq lnq2 lnw lnr, cost
mat b = e(b)
mat empinf_nhn = J(123,7,.)
forval i=1/123{
frontier lnc lnq lnq2 lnw lnr if _n!=`i', cost
mat b`i' = e(b)
forval j=1/5{
mat empinf_nhn[`i',`j'] = b[1,`j'] - b`i'[1,`j']
}
mat empinf_nhn[`i',6] = exp(1/2*b[1,6]) - exp(1/2*b`i'[1,6])
mat empinf_nhn[`i',7] = exp(1/2*b[1,7]) - exp(1/2*b`i'[1,7])
}
svmat double empinf_nhn

frontier lnc lnq lnq2 lnw lnr, cost d(e)
mat b = e(b)
mat empinf_nexp = J(123,7,.)
forval i=1/123{
capture noisily {
frontier lnc lnq lnq2 lnw lnr if _n!=`i', cost  d(e)
mat b`i' = e(b)
forval j=1/5{
mat empinf_nexp[`i',`j'] = b[1,`j'] - b`i'[1,`j']
}
mat empinf_nexp[`i',6] = exp(1/2*b[1,6]) - exp(1/2*b`i'[1,6])
mat empinf_nexp[`i',7] = exp(1/2*b[1,7]) - exp(1/2*b`i'[1,7])
}
}
svmat double empinf_nexp

rfrontier lnc lnq lnq2 lnw lnr, cost
mat b = e(b)
mat empinf_studhn = J(123,8,.)
forval i=1/123{
rfrontier lnc lnq lnq2 lnw lnr if _n!=`i', cost
mat b`i' = e(b)
forval j=1/5{
mat empinf_studhn[`i',`j'] = b[1,`j'] - b`i'[1,`j']
}
mat empinf_studhn[`i',6] = exp(b[1,6]) - exp(b`i'[1,6])
mat empinf_studhn[`i',7] = exp(b[1,7]) - exp(b`i'[1,7])
mat empinf_studhn[`i',8] = exp(b[1,8]) - exp(b`i'[1,8])
}
svmat double empinf_studhn
rename empinf_studhn7 ph
rename empinf_studhn8 empinf_studhn7
rename ph empinf_studhn8

rfrontier lnc lnq lnq2 lnw lnr, cost u(e)
mat b = e(b)
mat empinf_studexp = J(123,8,.)
forval i=1/123{
rfrontier lnc lnq lnq2 lnw lnr if _n!=`i', cost u(e)
mat b`i' = e(b)
forval j=1/5{
mat empinf_studexp[`i',`j'] = b[1,`j'] - b`i'[1,`j']
}
mat empinf_studexp[`i',6] = exp(b[1,6]) - exp(b`i'[1,6])
mat empinf_studexp[`i',7] = exp(b[1,7]) - exp(b`i'[1,7])
mat empinf_studexp[`i',8] = exp(b[1,8]) - exp(b`i'[1,8])
}
svmat double empinf_studexp
rename empinf_studexp7 ph
rename empinf_studexp8 empinf_studexp7
rename ph empinf_studexp8


rfrontier lnc lnq lnq2 lnw lnr, cost v(c)
mat b = e(b)
mat empinf_chn = J(123,8,.)
forval i=1/123{
rfrontier lnc lnq lnq2 lnw lnr if _n!=`i', cost v(c)
mat b`i' = e(b)
forval j=1/5{
mat empinf_chn[`i',`j'] = b[1,`j'] - b`i'[1,`j']
}
mat empinf_chn[`i',6] = exp(b[1,6]) - exp(b`i'[1,6])
mat empinf_chn[`i',7] = exp(b[1,7]) - exp(b`i'[1,7])
}
svmat double empinf_chn

rfrontier lnc lnq lnq2 lnw lnr, cost v(c) u(e)
mat b = e(b)
mat empinf_cexp = J(123,8,.)
forval i=1/123{
rfrontier lnc lnq lnq2 lnw lnr if _n!=`i', cost v(c) u(e)
mat b`i' = e(b)
forval j=1/5{
mat empinf_cexp[`i',`j'] = b[1,`j'] - b`i'[1,`j']
}
mat empinf_cexp[`i',6] = exp(b[1,6]) - exp(b`i'[1,6])
mat empinf_cexp[`i',7] = exp(b[1,7]) - exp(b`i'[1,7])
}
svmat double empinf_cexp



