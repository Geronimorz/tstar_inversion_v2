#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##Main function for t* inversion 
import numpy as np
from scipy.optimize import *
from scipy.linalg import lstsq
import tstar_parameters as tp
import tstar_load 
import tstarsub

def Loop_Record(orid, stalst, param, ARRIV, PGS, SGS):
    """
    Loop over each record, process the seismic information

    :param staP1lst:

    :param staP2lst:

    """
    saving={}   ## saved spectra: saving[sta][icase][POS]
    staP1lst=[]     ## STATIONS USED IN FINDING BEST fc AND alpha
    staP2lst=[]     ## STATIONS USED IN t*(P) INVERSION
    staP3lst=[]     ## STATIONS USED IN t*(P) INVERSION WITHOUT BAD FITTING
    staS1lst=[]     ## STATIONS USED IN FINDING BEST fc AND alpha (NOT USED)
    staS2lst=[]     ## STATIONS USED IN t*(S) INVERSION
    staS3lst=[]     ## STATIONS USED IN t*(S) INVERSION WITHOUT BAD FITTING
    for i, sta in enumerate(stalst):
        if sta=='AFI':
            continue
        if sta=='F01' or sta=='F02W':
            continue 
        ##yurong: question: are they bad stations?
        print('Working on station %s  (%d of %d)' % (sta,i+1,len(stalst)))
        chan = tstar_load.channels(sta)

        ## RETURN RAW DATA (REQUIRES P ARRIVAL, BUT NOT AN S ARRIVAL)
        ##    print("Reading seismic data from Antelope database")
        ## IF LAND STATION, dd(:,1)==TRANSVERSE && dd(:,2)==RADIAL
        ## IF OBS STATION, dd(:,1)==X && dd(:,2)==Y (x=EAST IF OBS HAS ORIENTATION IN SENSOR TABLE)
        (dd,tt,flag)=tstarsub.readseismo(param['pretime'],param['dt'],orid,sta,chan)
        if not flag:
            print('ERROR: Unable to read %s.%s' % (orid,sta))
            continue

        ## DETERMINE WHICH HORIZONTAL CHANNEL OF S ARRIVEL HAS LARGER AMPLITUDE
        if ARRIV[sta]['SDATA']:
            stiim=ARRIV[sta]['T1']-ARRIV[sta]['T0']   # S ARRIVAL IN dd SERIES
            indn=np.nonzero((tt>(stiim-1-param['WLS']))&(tt<stiim-1))
            ##yurong: questions: why -1 in NS direction?
            inds=np.nonzero((tt>stiim)&(tt<(stiim+param['WLS'])))
            snrew=np.absolute(dd[0][inds]).max()/np.absolute(dd[0][indn]).max()
            snrns=np.absolute(dd[1][inds]).max()/np.absolute(dd[1][indn]).max()
            if snrew > snrns:
                ARRIV.setdefault(sta,{})['schan'] = chan[0]
            else:
                ARRIV.setdefault(sta,{})['schan'] = chan[1]

        (p_dd,pn_dd)=tstarsub.fixwin(dd,tt,param,chan,ARRIV[sta],orid,sta)
        ## yurong: work on S wave
        PWINDATA  = np.vstack((p_dd,pn_dd))

        ## CALCULATE SPECTRA AND AUTO SELECTS FREQUENCY BAND ABOVE SET SNR
        ##======= 2 MEANS LOWER QUALITY DATA FOR t* INVERSION =======##
        (goodP2, goodS2, spec_px, freq_px, pspec, pn_spec, pfreq, pn_freq, frmin, frmax) \
            = tstarsub.dospec(PWINDATA, orid, sta, param, chan, 2)
        if not goodP2:
            print('No good P wave signal. Skip to next record.')
            continue

        ## SAVE SPECTRUM AND OTHER INFORMATION FOR EACH STATION
        saving[sta]={}
        saving[sta]['spec'], saving[sta]['nspec'] = [pspec], [pn_spec]
        saving[sta]['freq'], saving[sta]['nfreq'] = [pfreq], [pn_freq]
        saving[sta][1], saving[sta][2], saving[sta][3] = {}, {}, {}

        saving[sta]['Ptt'] = ARRIV[sta]['T0']
        saving[sta][2]['good']=[goodP2,goodS2]
        saving[sta][2]['frmin'], saving[sta][2]['frmax'] = frmin, frmax
        saving[sta][2]['p']=[freq_px,spec_px]
        # saving[sta][2]['s']=[freq_sx,spec_sx]

        ## CORRECTIONS OF GS
        correcP=float(PGS['gval'][PGS['stalist'].index(sta)])
        if correcP==0:
            print('Bad correction of P wave for station %s' % sta)
            continue
        saving[sta]['corr']=[correcP]
        staP2lst.append(sta)

        if ARRIV[sta]['SDATA'] and goodS2:
            correcS=float(SGS['gval'][SGS['stalist'].index(sta)])
            if correcS==0:
                print('Bad correction of S wave for station %s' % sta)
                continue
            saving[sta]['corr'].append(correcS)
            staS2lst.append(sta)
        ##======= 2 MEANS LOWER QUALITY DATA FOR t* INVERSION =======##

        if param['source_para'] == 1:   ## search for fc
            ##======= 1 MEANS HIGH QUALITY DATA FOR FINDING BEST fc AND alpha =======##        
            doplot=False
            # (spec_px,freq_px,spec_sx,freq_sx,spec,freq,n_spec,n_freq,frmn,frmx,
            # goodP1,goodS1)=tstarsub.dospec(PWINDATA,SWINDATA1,SWINDATA2,dt,
            #                 SDATA,orid,sta,snrcrtp1,snrcrts1,lincor,chan,doplot)
            (goodP1, goodS1, spec_px, freq_px, pspec, pn_spec, pfreq, pn_freq, frmin, frmax) \
                 = tstarsub.dospec(PWINDATA, orid, sta, param, chan, 1)
            if not goodP1:
                # print('No good P wave signal for finding best fc and alpha.')
                continue
            saving[sta][1]['good']=[goodP1,goodS1]
            saving[sta][1]['frmin']=frmin
            saving[sta][1]['frmax']=frmax
            saving[sta][1]['p']=[freq_px,spec_px]
            # saving[sta][1]['s']=[freq_sx,spec_sx]
            staP1lst.append(sta)

            if ARRIV[sta]['SDATA'] and goodS1 and goodS2:
                staS1lst.append(sta)
            ##======= 1 MEANS HIGH QUALITY DATA FOR FINDING BEST fc AND alpha =======##

    return staP1lst, staP2lst, staP3lst, staS1lst, staS2lst, staS3lst,\
        saving, ARRIV

def inversion(orid, saving, stalst, ORIG, POS, icase, param):
    """
        #################### INVERSION FOR t* ############################
    ##    EQUATION 3 IN Stachnik, Abers, Christensen, 2004, JGR     ##
    ##      d = Gm      (Nx1) = (Nx(M+1+)) ((M+1)x1)                ##
    ##      FOR A GIVEN fc AND alpha (GRID SEARCH):                 ##
    ##      d = [ln(A1)-ln(C1)+ln(1+(f1i/fc)**2),                   ##
    ##           ln(A2)-ln(C2)+ln(1+(f2i/fc)**2),                   ##
    ##           ln(AM)-ln(CM)+ln(1+(fMi/fc)**2)]                   ##
    ##      G = [[1, -pi*f1i*f1i**(-alpha), 0, ..., 0],             ##
    ##           [1, 0, -pi*f2i*f2i**(-alpha), ..., 0],             ##
    ##           [1, 0, 0, ..., -pi*fMi*fMi**(-alpha)]]             ##
    ##      m = [[ln(Moment)],[tstar01],[tstar02],...,[tstar0M]     ##
    ##################################################################
 
    """

    idata = (icase==3 and 2 or icase)
    data = tstarsub.buildd(saving,stalst,ORIG,POS,idata,param['source_para'])
    G = tstarsub.buildG(saving,stalst,param['alpha'],POS,idata,param['source_para'])
    Ginv=np.linalg.inv(np.dot(G[:,:,0].transpose(),G[:,:,0]))
    model,residu=nnls(G[:,:,0],data[:,0])
    if param['source_para'] == 1:
        lnmomen=model[0]      ## MOMENT
        tstar=model[1:]       ## t*
    else:
        tstar=model           ## t*
    if icase == 2:
        ferr=open(tp.resultdir+'/%s_perr%03d.dat' % (orid,int(param['alpha']*100)),'w')
        ferr.write('%15f %7d %15f %15f\n' % (residu,data.shape[0],
                                        (residu**2)/np.sum(data[:,0]**2),
                                        (residu/np.sum(data[:,0]))))
        ferr.close()

    ## ESTIMATE MOMENT ERROR BASED ON ALL DATA VARIANCES
    vardat=residu/np.sqrt(data.shape[0]-2)
    # lnmomenPerr=np.sqrt(vardatP2*GP2inv[0][0])
    ## ESTIMATE t* ERRORS BASED ON DATA VARIANCES FOR EACH t*
    estdata=np.dot(G[:,:,0],model)

    k1=0
    for ista in range(len(stalst)):
        sta = stalst[ista]
        ndat=len(saving[sta][idata][POS.lower()][0])
        k2=k1+ndat
        dat=data[k1:k2]
        est=estdata[k1:k2]
        var=(np.linalg.norm(dat-est)**2)/(ndat-2)    ## POSTERIOR VARIANCE USED AS PRIOR VARIANCE
        saving[sta][icase]['tstar']=[tstar[ista]]
        saving[sta][icase]['misfit']=[np.sqrt(var*(ndat-2))/ndat]    
        if param['source_para'] == 1: ## grid search for Mw, one more list for G matrix
            saving[sta][icase]['err']=[np.sqrt(var*Ginv.diagonal()[ista+1])] ## cov(m)=cov(d)inv(G'G) FOR OVERDETERMINED PROBLEM
        else:
            saving[sta][icase]['err']=[np.sqrt(var*Ginv.diagonal()[ista])] ## cov(m)=cov(d)inv(G'G) FOR OVERDETERMINED PROBLEM        saving[sta][icase]['aveATTEN']=[(1000*tstar[ista]/saving[sta]['Ptt'])]
        saving[sta][icase]['aveATTEN']=[(1000*tstar[ista]/saving[sta]['Ptt'])]
        if icase == 2:
            ## Measure how synthetic curve fit the observed data
            if saving[sta][icase]['good'][0]:
                pfitting=tstarsub.fitting(saving,sta,ORIG,POS,param['alpha'],2)
                saving[sta][icase]['fitting']=[pfitting]
            else:
                saving[sta][icase]['fitting']=[1000]
        elif icase == 3:
            lnM = np.log(ORIG['mo'])
            saving[sta][icase]['resspec']=[tstarsub.calresspec(saving[sta],POS,lnM,
                                                ORIG['fc'],param['alpha'])]
        k1=k2

    return saving

def output_results(orid, staP3lst, param, ORIG, saving):
    ## OUTPUT P RESIDUAL SPECTRA FOR SITE EFFECTS
    for sta in staP3lst:
        sitefl=tp.resultdir+'/%s_Presspec_%s.dat' % (orid,sta)
        np.savetxt(sitefl,saving[sta][3]['resspec'][0], fmt='%10.4f  %15.8e  %6.2f')

    ## OUTPUT RESULTS FOR TOMOGRAPHY
    ftstar=open(tp.resultdir+'/%s_pstar%03d.dat' % (orid,int(param['alpha']*100)),'w')
    for sta in staP3lst:
        ftstar.write('%s  %.4f  %.4f  %.4f  %f  %f  %f  %.2f\n' %
                     (sta,
                      ORIG['lat'],
                      ORIG['lon'],
                      ORIG['dep'],
                      saving[sta][3]['tstar'][0],
                      saving[sta][3]['err'][0],
                      saving[sta][3]['misfit'][0],
                      saving[sta][3]['aveATTEN'][0]))
    ftstar.close()

    ## PLOT P SPECTRUM FOR EACH STATION
    if param['doplotspec']:
        for sta in staP3lst:
            if saving[sta][2]['good'][0]:
                print('Plotting P spectrum of ' + sta)
                lnM = np.log(ORIG['mo'])
                tstarsub.plotspec(saving[sta],sta,orid,'P',lnM,
                                  ORIG['fc'],param['alpha'],3)

    return
