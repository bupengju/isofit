from __future__ import absolute_import, division, print_function

import os, sys, glob, argparse
from os.path import splitext, exists as pathexists, join as pathjoin, \
  split as pathsplit


import matplotlib
matplotlib.use('agg')
import numpy.random as random
import scipy as s
import pylab as pl
pl.ioff()
import numpy as np # BB: I need my np.* alias!!!

from scipy.io import loadmat,savemat

from sklearn.base import clone
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from sklearn.neural_network import MLPRegressor as MLP
from sklearn.externals import joblib

from copy import deepcopy
from time import sleep, time as gettime

# Need some specific environment variables and libraries
requisites = ['ISOFIT_BASE']
for req in requisites:
 if os.getenv(req) is None:
   print('Please define the environment variable: '+req)
   sys.exit()

from common import *
  
def mad(a,axis=0,medval=None,unbiased=False):    
  '''
  computes the median absolute deviation of a list of values
  mad = median(abs(a - medval))/c
  '''
  # try:
  #     from statsmodels.robust.scale import mad as _mad
  #     center = medval or np.median
  #     c = 0.67448975019608171 if unbiased else 1.0
  #     return _mad(a, c=c, axis=axis, center=center)
  # except:
  ma = np.asarray(a)
  medval = medval or np.median(ma)
  return np.median(np.abs(ma-medval))

def plotmeanstd(ax,wl,meanv,stdv,diff=False):
  minv = meanv-stdv
  maxv = meanv+stdv
  if diff:
    meanp = s.diff(meanv)
    minv,maxv = s.diff(minv),s.diff(maxv)
    wvl = wl[:-1]
    stdp = s.diff(meanv+stdv)
  else:
    wvl = wl
    meanp = meanv
    stdp = stdv

  ax.plot(wvl,minv,'b')
  ax.plot(wvl,meanp,'r')
  ax.plot(wvl,maxv,'b')

def normalize_inputs(inputs,names,clip=False):
  normf = dict(phi=(0.0,s.pi), umu=(0.5,1.0), aod=(0.0,0.5),
               h2o=(0.0,2.5),  albedo=(0.0,1.0))

  assert(inputs.shape[1]==len(names))
  outputs = inputs.copy()  
  for i,namev in enumerate(names):
    name = namev.lower()
    if name in normf:
      vmin,vmax = normf[name]
    elif name.startswith('ao'):
      vmin,vmax = normf['aod']
    elif 'h2o' in name:
      vmin,vmax = normf['h2o']
    elif name.startswith('alb'):
      vmin,vmax = normf['albedo']
    else:
      raise Exception('Unknown input: '+namev)
    outputs[:,i] = (outputs[:,i]-vmin) / (vmax-vmin)

  if outputs.min() < 0 or outputs.max() > 1:
      if clip:
        warn('clipping normalized inputs outside the [0,1] range')
        outputs = s.clip(outputs,0.0,1.0)
      else:
        warn('normalized inputs exist that are outside [0,1] range')

  return outputs

def copy_weights(model_dst,model_src):
  # initialize internal sklearn state by training for a single epoch on random inputs
  input_dim = model_src.coefs_[0].shape[0]
  output_dim = model_src.coefs_[-1].shape[-1]

  model_dst_params = model_dst.get_params(deep=True)
  model_dst_params['learning_rate_init'] = model_src._optimizer.learning_rate  
  model_params = model_dst_params.copy()
  model_params['max_iter'] = 1
  model_params['warm_start'] = False
  # use learning rate of model_src
  model_dst.set_params(**model_params)

  rand_inputs = np.random.rand(200,input_dim)
  rand_tgts = np.random.rand(200,output_dim)
  model_dst.fit(rand_inputs,rand_tgts)

  # copy weights/biases from model_src to model_dst
  #print('model_src.coefs_[0].shape: "%s"'%str((model_src.coefs_[0].shape)))
  #print('model_src.coefs_[-1].shape: "%s"'%str((model_src.coefs_[-1].shape)))
  model_dst.coefs_ = model_src.coefs_
  model_dst.intercepts_ = model_src.intercepts_


  model_dst.set_params(**model_dst_params)

  # pretend we're starting from scratch
  model_dst.n_iter_ -= 1
    
  return model_dst

def fit(model, model_file, train_inputs, train_tgts, val_inputs, val_tgts,
        max_iter, es_tol=-1, val_step=25, prev_model_file=None,
        shuffle_train=False, verbose=False):
  best_iter,best_mse,best_mae = -1,np.inf,np.inf
  tr_inputs,tr_tgts = train_inputs,train_tgts

  model_fit = clone(model)
  model_params = model.get_params(deep=True)
  model_params['tol'] = es_tol
  model_params['early_stopping'] = (es_tol != -1)

  # always use warm start since model_fit should already be initialized
  model_params['warm_start'] = True
  model_params['verbose'] = verbose

  for val_epoch in range(0,max_iter,val_step):
    if shuffle_train:
      del tr_inputs,tr_tgts
      shuffle_idx = np.random.permutation(len(tr_inputs))
      tr_inputs = train_inputs[shuffle_idx].copy()
      tr_tgts = train_tgts[shuffle_idx].copy()

    max_iter_fit = val_epoch+val_step
    if val_epoch==0:
      # initialize model on first epoch
      if prev_model_file:
        # initialize current model using prev_model_file
        model_prev = joblib.load(prev_model_file)
        model_fit = copy_weights(model_fit,model_prev)
            
    # manually validate every val_step iterations        
    model_params['max_iter'] = val_step
    model_fit.set_params(**model_params)
    #print('model_params: "%s"'%str((model_params)))
    #model_fit_params = model_fit.get_params(deep=True)
    #print('model_fit_params: "%s"'%str((model_fit_params)))
    model_fit.fit(tr_inputs,tr_tgts)
    #print('model_fit.n_iter_: "%s"'%str((model_fit.n_iter_)))
    #print('model_fit.max_iter: "%s"'%str((model_fit_params['max_iter'])))
    
    val_preds = model_fit.predict(val_inputs).reshape(val_tgts.shape)
    val_mse = mse(val_tgts,val_preds)
    statestr = ['epoch %d: val_mse=%.16f (prev best=%.16f)'%(val_epoch,val_mse,best_mse)]
    model_iter = model_fit.n_iter_

    #print('val_epoch,max_iter_fit,model_iter: "%s"'%str((val_epoch,max_iter_fit,model_iter)))
    #raw_input()
    
    # save model with smallest loss
    if val_mse < best_mse:
      statestr += ['epoch %d: new best val_mse=%.16f (prev best=%.16f)'%(val_epoch,val_mse,best_mse)]
      model_best = model_fit
      best_mse = val_mse
      best_mae = mae(val_tgts,val_preds)
      joblib.dump(model_fit, model_file)
      model_filename = pathsplit(model_file)[1]
      statestr += ['epoch %d: saved model to %s'%(val_epoch,model_filename)]
    
    stop_early=False
    if model_iter < val_epoch+val_step:
      # model converged in fit() between val_epoch and val_epoch+val_step
      stop_early = True
      statestr += ['epoch %d: stopping early at model_iter %d'%(val_epoch,model_iter)]

    if len(statestr) != 0:
      print('\n'.join(statestr))

    if stop_early:
      break

  print()
  model_best = joblib.load(model_file)
  return model_best, (best_mse, best_mae)

def main():

  # pull in the hyperparameters for the neural network
  parser = argparse.ArgumentParser()
  parser.add_argument('config_file')
  parser.add_argument('--retrain',help='retrain model',action='store_true')
  parser.add_argument('--val_only',help='generate validation output only',action='store_true')
  parser.add_argument('--train_fullres',help='train on full resolution spectra',action='store_true')
  parser.add_argument('--disable_wp',help='disable weight propagation',action='store_true')

  args       = parser.parse_args()
  retrain    = args.retrain
  use_wp     = (not args.disable_wp)
  val_only   = args.val_only
  full_res   = args.train_fullres

  configname     = pathsplit(args.config_file)[1].replace('.json','')
  config         = json_load_ascii(args.config_file)
  wavelengthfile = config["wavelength_file"]
  names          = config['inputvector_order']
  trainfile      = config['trainfile']
  trainfile_rs   = config.get('trainfile_rs',None)

  if 'alb' not in names:
    names.append('alb')

  # assume outdir is always the same as trainfile (not trainfile_rs)
  outdir,trainf = pathsplit(trainfile)
  
  wl_inst, fwhm_inst = load_wavelength_file(wavelengthfile)

  # prm20151026t173213_libRadtran.mat (full res) shape = (9072, 7101)
  # prm20151026t173213_libRadtran_PRISM_rs.mat (instrument) shape = (9072, 246)

  if full_res:
    # Train on full-resolution RTM channels, rather than (downsampled)
    # instrument channels
    print('loading trainfile: "%s"'%str((trainfile)))
    D = loadmat(trainfile)
    wl =  D['wl'].squeeze()
    assert(len(wl)>len(wl_inst))
  else:
    # Load the resampled instrument channels
    print('loading trainfile_rs: "%s"'%str((trainfile_rs)))
    D  = loadmat(trainfile_rs)
    wl =  D['wl'].squeeze()
    assert(len(wl)==len(wl_inst))
    assert((wl==wl_inst).all())
    
  inputs = s.float32(D['input'])
  print('inputs.shape: "%s"'%str((inputs.shape)))
    
  tgts = D['rho']
  
  # dimensionality of state space = n_inputs
  n_inputs = len(names)
  print('names: "%s"'%str((names)))
  assert(n_inputs == inputs.shape[1])
  
  n_samples = len(inputs)
  n_wl = len(wl)

  # construct train/test partitions
  random.seed(42)    
  samp_idx = s.arange(n_samples)    
  tr_mask = np.zeros(n_samples,dtype=np.bool8)
  stratify_inputs=True
  if stratify_inputs:
    uniq = []
    trinputs,teinputs = [],[]
    for i in range(n_inputs):
      # hold out central unique value of each variable, train on rest
      holdi = np.unique(inputs[:,i])
      teinputs.append(np.array([holdi[len(holdi)//2]]))
      trinputs.append(np.setdiff1d(holdi,teinputs[-1]))
      print('i,trinputs[i],teinputs[i]: "%s"'%str((i,trinputs[i],
                                                   teinputs[i])))
      uniq.append(holdi)
    
    # NOTE: leave all albedo values in training set
    # partition on remaining (4) states
    for i in range(n_inputs):
      if not names[i].startswith('alb'):
        tr_mask |= np.isin(inputs[:,i],teinputs[i])
    # invert mask to get training indices 
    tr_mask = ~tr_mask
  else:
    #  validate on (100*p)% of the data
    p = 0.2
    random.shuffle(samp_idx)
    tr_mask[:int(n_samples*(1-p))] = 1
    
  tr_idx = samp_idx[tr_mask]
  val_idx = samp_idx[~tr_mask]
  n_samplesv = len(val_idx)
  print('n_samples: %d'%(n_samples))
  print('n_train:   %d'%(len(tr_idx)))
  print('n_val:     %d'%(n_samplesv))
  print('n_inputs:  %d'%(n_inputs))
  print('n_wl:      %d'%(n_wl))
  
  # initialize model
  # n_layers = n_hidden_layers + 1 (output)
  n_layers = 2
  n_hidden = 55
  if n_layers==2:
    weight_labs = ['input','output']
    layers = (n_hidden,)
  else:
    weight_labs = ['input','hidden','output']
    layers = (n_hidden,n_hidden,)

  # halfwidth of overlapping channel input range
  # n_half == 0 -> monochromatic
  # n_half == 1->3 channel averaging, 2->5 channel averaging, ...
  n_half = 0
  average_over = True # only valid if n_half > 0
  n_over = 2*n_half+1

  # 'auto'== 200 samples/batch
  batch_size = 'auto' 

  # train first subnetwork for many more epochs to ensure convergence
  init_max_iter = 500
  max_iter = 500

  long_train = False
  if long_train:
    init_max_iter *= 4
    max_iter *= 4

  # compute validation accuracy every val_step epochs
  init_step = 100
  val_step = 25

  # early stopping for val models, init via weight propagation
  es_val = True
  # early stopping for final models, initialized with val weights
  es_fin = True 

  # tol == -1 -> train until max_iter 
  tol_init = -1 
  tol_val = 1e-200 if es_val else -1
  tol_fin = 1e-200 if es_fin else -1

  # use a small percentage of training data for early stopping
  es_percent = 0.1 

  # set early stopping=True and disable on case-by-case basis
  mlpparms = dict(hidden_layer_sizes=layers, activation='relu', solver='adam',
                  alpha=1e-5, batch_size=batch_size, learning_rate='adaptive',
                  learning_rate_init=0.001, power_t=0.5, max_iter=max_iter,
                  random_state=42, early_stopping=True, tol=tol_init,
                  warm_start=False, momentum=0.9, nesterovs_momentum=True,
                  shuffle=False, validation_fraction=es_percent,
                  beta_1=0.9, beta_2=0.999, epsilon=1e-10, verbose=False,
                  n_iter_no_change=10)
  model = MLP(**mlpparms) 

  # map state parameter values to unit range
  inputs = normalize_inputs(inputs,names,clip=False)
  
  val_loss = dict(mse=[],mae=[])

  # pick a few validation toa spectra at varying mad to show in detail
  val_tgts = tgts[val_idx]
  val_dtgts = np.diff(val_tgts,1)
  val_dtgts_med = np.median(val_dtgts,axis=0)
  val_dtgts_mad = abs(val_dtgts-val_dtgts_med).sum(axis=1)
  sorti = np.argsort(val_dtgts_mad)
  val_qper = [0.25,0.5,0.75]
  val_toaidx = [sorti[int(n_samplesv*qp)] for qp in val_qper] 
  val_toalab = ['argq%d'%(qi*100) for qi in val_qper]
  val_toalab = dict(zip([sorti[0]]+val_toaidx+[sorti[-1]],
                        ['argmin']+val_toalab+['argmax']))
  for ridx,rlab in val_toalab.items():
    print('val_dtgts_mad[%s]: "%s"'%(rlab,str((val_dtgts_mad[ridx]))))
    print('inputs[%s]: "%s"'%(rlab,str((inputs[val_idx[ridx]]))))

  # bookkeeping for selected val_toa spectra + predictions
  val_toatrue = dict([(idx,s.zeros(n_wl)) for idx in val_toaidx])
  val_toapred = dict([(idx,s.zeros(n_wl)) for idx in val_toaidx])
  val_toamse = dict([(idx,s.zeros(n_wl)) for idx in val_toaidx])

  plot_toadat=False
  if plot_toadat:
    dwl=wl[:val_dtgts.shape[1]]
    fig0,ax0 = pl.subplots(2,1,sharex=True,sharey=False)
    for ridx,rlab in val_toalab.items():
      ax0[0].plot(wl,val_tgts[ridx],label=rlab)
      ax0[1].plot(dwl,val_dtgts[ridx],label='diff(%s)'%rlab)

    ax0[0].legend()
    ax0[1].legend()
    pl.show()
  
  accum = s.zeros(n_wl)

  modelclass =  '_'.join(['mlp']+list(map(str,layers))+[str(n_over)])
  if long_train:
    modelclass += '_longtr'
  if not use_wp:
    modelclass += '_nowp'
    
  modeldir = pathjoin(outdir,modelclass)
  print('modeldir: "%s"'%str((modeldir)))  
  if not pathexists(modeldir):
    os.makedirs(modeldir)

  log_filename = 'train_pid%s.log'%str(os.getpid())
  log_file = pathjoin(modeldir,log_filename)
  print('Writing log_file=%s'%log_file)
  sleep(1)

  log_fid = open(log_file,'w')
  print('# c, iter, mse, mae, time',file=log_fid)
      
  modelprefix = pathjoin(modeldir,splitext(trainf)[0])
    
  modelbase = modelprefix+'_c%s.pkl'
  init_modelbase = modelprefix+'_init_c%s.pkl'
  fin_modelbase = modelprefix+'_fin_c%s.pkl'

  # weight/bias data bookeeping
  W,b = {},{}
  
  # weight/bias output files
  Wf,bf = {},{}
  
  # validation weights/biases constructed on training set,
  # assesed on validation set  
  W['val'] = [np.zeros([n_wl,n_inputs,n_hidden]),
              np.zeros([n_wl,n_hidden,n_over])]
  b['val'] = [np.zeros([n_wl,n_hidden]),
              np.zeros([n_wl,n_over])]
  
  # "final" weights/biases constructed using *all* states
  # NOTE: use these during deployment to isofit/rt_nn.py
  W['fin'] = [np.zeros([n_wl,n_inputs,n_hidden]),
              np.zeros([n_wl,n_hidden,n_over])]
  b['fin'] = [np.zeros([n_wl,n_hidden]),
              np.zeros([n_wl,n_over])]

  # output file paths for validation/final outputs
  # TODO (BDB, 04/17/19): make multi_layer consistent 
  Wf['val'] = [modelprefix+'_W%dv.npy'%l for l in [1,2]]
  bf['val'] = [modelprefix+'_b%dv.npy'%l for l in [1,2]]
  
  Wf['fin'] = [modelprefix+'_W%dnpy'%l for l in [1,2]]
  bf['fin'] = [modelprefix+'_b%d.npy'%l for l in [1,2]]

  if n_layers==3:
    Wf['val'] = [Wf['val'][0],modelprefix+'_Wmv.npy',Wf['val'][1]]
    bf['val'] = [bf['val'][0],modelprefix+'_bmv.npy',bf['val'][1]]
    Wf['fin'] = [Wf['fin'][0],modelprefix+'_Wm.npy',Wf['fin'][1]]
    bf['fin'] = [bf['fin'][0],modelprefix+'_bm.npy',bf['fin'][1]]
    
    W['val'] = [W1v[0],np.zeros([n_wl,n_hidden,n_hidden]),W2v[1]]
    b['val'] = [b1v[0],np.zeros([n_wl,n_hidden]),b2v[1]]
    W['fin'] = [W1f[0],np.zeros([n_wl,n_hidden,n_hidden]),W2f[1]]
    b['fin'] = [b1f[0],np.zeros([n_wl,n_hidden]),b2f[1]]
  
  abserr = dict(fin=np.zeros([n_samples,len(wl)]), val=np.zeros([n_samplesv,len(wl)]))
  sqderr = dict(fin=np.zeros([n_samples,len(wl)]), val=np.zeros([n_samplesv,len(wl)]))

  # initialize model on initial channel(s)
  cr = s.arange(0,n_over)

  tr_input = inputs[tr_idx]
  val_input = inputs[val_idx]
  tr_tgts = tgts[tr_idx,cr]
  val_tgts = tgts[val_idx,cr]

  init_model_file = init_modelbase%str(0)
  model_params = model.get_params(deep=True)
  model_init = None
  model_init_time = gettime()
  if not pathexists(init_model_file) or retrain:
    fit_init = fit(model, init_model_file, tr_input, tr_tgts, val_input,
                   val_tgts, init_max_iter, es_tol=tol_init,
                   val_step=init_step)
                   
    model_init,model_init_err = fit_init
    model_init_mse,model_init_mae = model_init_err
    model_init_time = gettime()-model_init_time
    model_init_iter = model_init.n_iter_
    print('%d, %d, %.16f, %.16f, %d'%(-1,model_init_iter,model_init_mse,
                                      model_init_mae,model_init_time),
          file=log_fid)
    log_fid.flush()
  else:
    model_init = joblib.load(init_model_file)
    print('loaded',init_model_file)
    
  models = [model_init]
  model_file = modelbase%str(0)
  
  # train channelwise subnetworks
  for c in range(n_wl):    
    # define channel range (cmin==cmax for monochromatic)
    n_off = n_wl-c
    if c < n_half+1:
      cmin,cmax = 0,n_over-1
    elif n_off < n_half+1:
      cmin,cmax = n_wl-n_over,n_wl-1
    else:
      cmin,cmax = c-n_half,c+n_half
      
    cr = s.arange(cmin,cmax+1) if (n_half==0 or average_over) else c
    
    print('\n##### Training subnetwork for center channel %d, wl=[%.2f,%.2f] ##########'%(c,wl[cmin],wl[cmax]))
    # grab the training/validation targets for the current channel(s)  
    fin_tgts = tgts[:,cr].reshape([-1,n_over])
    tr_tgts = tgts[tr_idx,cr].reshape([-1,n_over])
    val_tgts = tgts[val_idx,cr].reshape([-1,n_over])

    # update model file after storing the previous file
    prev_model_file = None
    if use_wp:
      prev_model_file = model_file if c!=0 else init_model_file
    model_file = modelbase%str(c)

    model_c_time = gettime()
    if not pathexists(model_file) or retrain:
      # train new model, save to model_file
      model_c = clone(model)      
      model_c.set_params(**model_params)
      fit_c = fit(model, model_file, tr_input, tr_tgts, val_input,
                  val_tgts, max_iter, val_step=val_step,
                  es_tol=tol_val, prev_model_file=prev_model_file)
      model_c,model_c_err = fit_c
      model_c_mse,model_c_mae = model_c_err
      model_c_iter = model_c.n_iter_
      model_c_time = gettime()-model_c_time
    else:
      # restore from model_file
      model_c = joblib.load(model_file)
      val_preds = model_c.predict(val_input).reshape(val_tgts.shape)
      model_c_iter = model_c.n_iter_
      model_c_mse = mse(val_tgts,val_preds)
      model_c_mae = mae(val_tgts,val_preds)
      model_c_time = gettime()-model_c_time

    print('%d, %d, %.16f, %.16f, %d'%(c,model_c_iter,model_c_mse,model_c_mae,
                                      model_c_time),file=log_fid)
    log_fid.flush()

    if c==0 and model_init is not None:
      # replace initial model with refined version
      models[0] = model_c
    else:
      models.append(model_c)

    # save the output validation weights / biases
    for l in range(n_layers):
      W['val'][l][c] = np.array(model_c.coefs_[l])
      b['val'][l][c] = np.array(model_c.intercepts_[l])


    val_preds = model_c.predict(val_input).reshape(val_tgts.shape)
    #print('cr,val_preds.shape,val_tgts.shape: "%s"'%str((cr,val_preds.shape,
    #                                                    val_tgts.shape)))
    
    # generate detailed summaries for val_toaidx spectra
    for ri in val_toaidx:
      if n_half==0 or average_over:
        # model generates predictions for 0 or more adjacent channels and
        # we average the predictions for all generated channels
        predi,truei = val_preds[ri],val_tgts[ri]
      else:
        # model generates predictions for 1 or more adjacent channels but
        # we only consider the target channel in generating output
        # target channel centered unless c first/prev channel
        predi,truei = val_preds[ri][c-cmin],val_tgts[ri][c-cmin]

      val_toaabserri = np.abs(predi-truei)
      val_toatrue[ri][cr] = truei
      val_toapred[ri][cr] += predi
      val_toamse[ri][cr] += (val_toaabserri*val_toaabserri)

    # keep track of how many times we generate a prediction for each channel
    # (accmum==ones(n_wvl) for monochromatic case)
    accum[cr] += 1
    val_abserr = np.abs(val_tgts-val_preds)
    val_sqderr = val_abserr*val_abserr

    sqderr['val'][:,cr] += val_sqderr
    abserr['val'][:,cr] += val_abserr

    # track the mean/std/median/mad of the mse and mae
    val_loss['mse'].append([np.mean(val_sqderr),np.std(val_sqderr),
                            np.median(val_sqderr),mad(val_sqderr)])
    val_loss['mae'].append([np.mean(val_abserr),np.std(val_abserr),
                            np.median(val_abserr),mad(val_abserr)])

    if val_only: # skip training "final" model
      print('val_only==True, skipping production model fit')
      continue

    print('\n##### Training full subnetwork for center channel %d, wl=[%.3f,%.3f] ##########'%(c,wl[cmin],wl[cmax]))
    fin_model_file = fin_modelbase%(str(c))
    fin_model_filename = pathsplit(fin_model_file)[1]
    model_f_time = gettime()
    if not pathexists(fin_model_file) or retrain:
      # train "final" production model on *all* inputs
      # start with converged validation model for this channel
      model_f = deepcopy(model_c)
      model_f.fit(inputs, fin_tgts)
      model_f_time = gettime()-model_f_time
      joblib.dump(model_f, fin_model_file)
      print('saved',fin_model_filename)
    else:
      model_f = joblib.load(fin_model_file)
      print('loaded',fin_model_filename)

    fin_preds = model_f.predict(inputs).reshape(fin_tgts.shape)
    fin_abserr = np.abs(fin_tgts-fin_preds)
    fin_sqderr = fin_abserr*fin_abserr
    sqderr['fin'][:,cr] += fin_sqderr
    abserr['fin'][:,cr] += fin_abserr

    # extract the final model weights / biases
    for l in range(n_layers):
      W['fin'][l][c] = np.array(model_f.coefs_[l])
      b['fin'][l][c] = np.array(model_f.intercepts_[l])

  # end of channelwise training loop
  
  evalkeys = ['val'] if val_only else ['val','fin']
  for evalkey in evalkeys:
    abserr[evalkey] = abserr[evalkey] / accum 
    sqderr[evalkey] = sqderr[evalkey] / accum 

    outmat = modelprefix+'_%s_sqderr.mat'%evalkey
    savemat(outmat,{'training_idx':tr_idx,
                    'validation_idx':val_idx,
                    'sqderr':sqderr[evalkey],
                    'abserr':abserr[evalkey],
                    'wl':wl})
    print('saved',outmat)
    Woutf,boutf = Wf[evalkey],bf[evalkey]
    Wout,bout = W[evalkey],b[evalkey]    
    for l,(Wl,bl) in enumerate(zip(Woutf,boutf)):
      np.save(Wl,Wout[l])
      np.save(bl,bout[l])
      
    print('saved %s weights to: "%s"'%(evalkey,str(((Woutf[0],boutf[0]),(Woutf[1],boutf[1])))))
    
  # plot the selected val_toaidx predictions vs. actuals
  for idx in val_toaidx:
    toatitle = ', '.join('%s=%.2f'%(n,v) for n,v in zip(names,inputs[idx]))

    toatrue = val_toatrue[idx]
    toapred = val_toapred[idx] / accum
    toamse  = val_toamse[idx] / accum

    toafig = modelprefix+'_toa%d_%s_pred.pdf'%(val_idx[idx],
                                               val_toalab[idx])
    fig,ax = pl.subplots(3,1,sharex=True,sharey=False)
    ax[0].plot(wl,toatrue)
    ax[0].plot(wl,toapred,c='r',ls=':')
    ax[1].plot(wl[:-1],s.diff(toatrue))
    ax[1].plot(wl[:-1],s.diff(toapred),c='r',ls=':')
    ax[2].plot(wl,toamse)
    pl.suptitle(toatitle)
    pl.savefig(toafig)
    print('saved toafig: "%s"'%str((toafig)))

  # plot mse/mae error curves    
  for errkey in val_loss:
    errdat = np.array(val_loss[errkey])
    errmean,errstd = errdat[:,0],errdat[:,1]
    errstr='mean val_%s: %g, std: %g'%(errkey,s.mean(errdat[:,0]),s.std(errdat[:,0]))
    print(errstr)
  
    errfig = modelprefix+'_%s.pdf'%errkey
    fig,ax = pl.subplots(2,1,sharex=True,sharey=False)
    plotmeanstd(ax[0],wl,errmean,errstd)
    plotmeanstd(ax[1],wl,errmean,errstd,diff=True)  
    ax[0].set_ylabel(errkey)
    ax[1].set_ylabel('diff(%s)'%errkey)
    ax[0].set_title(errstr)
    pl.savefig(errfig)
    print('saved errfig: "%s"'%str((errfig)))
    
  # free some memory
  del D
          
if __name__ == '__main__':
    main()
