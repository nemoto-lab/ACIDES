"""
This is a module file for ACIDES, including
- ACIDES class
  # __init__                           
    to define parameters, etc.
  # Exp_model_fitting  
    This function performs the first part of the two-step alrogithm (please refer to the method section of the manuscript.)
      Input data  : count data for each variant, alpha and beta
      Output data : a and b of each variant, theta and the expected value of counts.
  # r_fitting_from_lambda_and_counts
    This function performs the second part of the two-step alrogithm (please refer to the method section of the manuscript.)
      Input data  : count data, and its expected value (predicted by the model) for each variant, alpha and beta (This is for initial parameters.)
      Output data : a and b of each variant, theta and the expected value of counts.
  # making_synthetic_data
    This function generates synthetic data based on lambda (expected value of counts), alpha, beta.
      Input data  : lambda (expected value of counts), alpha, beta
      Output data : count data generated synthetically
  # thresholding_data
    This function trims count data using a pre-determined threshold.
      Input data  : count data
      Output data : trimmed count data
  # initial_condition_for_aa_bb
    The initial parameter of a is determined as (count[last time] - count[initial time]) / (last time - initial time).
    log(rho0) is determined from "a" as an intercept. 
    If replicates are avialble, average it over replicate. 
      Input data  : count data, the total NGS reads for each round, time point data
      Output data : initial parameters for a and log(rho0)
  # fit
    a function to infer alpha, beta, "a" and log(rho0) from count data  ############
      Input data  : count data
      Output data : self 
  # fit_after_fixing_parameters
    After determining alpha and beta with theta, we detemine a and log(rho0) for all the sequences, including those that were exluded (due to Untilwhichvalue_).
    This function can be available only after using "fit" function. 
      Output data : self
  # lambda_when_two_time_dataset
    This function is to estimate lambda when the data has only two time points. In this case, we don't need "Exp_model_fitting".
      Output: lambda (expected value of counts). This is estimated as the average count over all the replicates.
  # rr_removeoutside
    This function is to calculate the value of r from a given value of lambda with a fitting function.
      Input: lambda, fitting function
      Output: r
  # fit_for_two_time_dataset
    This function is to infer alpha, beta, "a" and log(rho0) from count data 
      Input data  : count data
      Output data : obtained alpha and beta for t=1
                    obtained alpha and beta for t=0
                    lambda values (the expected value of counts)
                    
- ACIDES_FIGURE_AVERAGE class
  # __init__
    to define parameters, etc.
  # Experimental_reliability_
    A function to compute RR. 
    For the definition of RR, please refer to the Method section of the manuscript. 
  # ranking_probability_bootstrap
    This is a function to estimate the confidence interval of the corrected rank based on 
    a sparse matrix with (columns: mapping from the original naive rank to the resampled naive rank) and (rows: resampled sample's index)
  # Ranking_probability_based_on_ai
    This is a function to draw the corrected rank (x-axis is the naive rank, y-axis is the corrected rank.)

Please don't hesitate to reach out for me if you have any question: nemoto.takahiro.prime@osaka-u.ac.jp
"""

__version__ = "0.0.0"
__author__ = ["Takahiro Nemoto"]
__all__ = ["ACIDES"]

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib
font = {'size'   : 20}
matplotlib.rc('font', **font)
import pickle
from scipy import sparse 
from scipy.stats import nbinom
from scipy.optimize import minimize   
from scipy.special import polygamma
from importlib import reload
import sys
from scipy.stats import poisson
from scipy.special import gamma
from numpy.linalg import inv
from scipy.optimize import curve_fit
from scipy import special
from sklearn.metrics import r2_score
from matplotlib.pyplot import figure

from scipy.interpolate import interp1d
from scipy.stats import norm

from constrained_linear_regression import ConstrainedLinearRegression

class ACIDES():
  def __init__(
          self,
          Inference_type,                     # "Negbin": inference with negative binomial distribution, "Pois": inference with Poisson distribution
          theta_term_yes,                     # "yes": using theta term for normalization, "no": not using it.
          t_rounds,                           # the time at which NGS is taken (numpy array)
          folder_name = '.',                  # the path to the folder to save the data (oject)
          random_num=None,                    # random seed (integer or None). 
          Regularization_ = 0,                # regularization parameter
          save_data = 'yes',                  # "yes": save data
          BiasCorrectionwith = 'Yes_inverse', # "Yes": with a bias correction, "No": without a bias correction. "Yes_inverse"" ; with a bias correction and with the inference with 1/r 
          Num_division = 100,                 # how many bins when fitting r(lambda).
          Untilwhichvalue_= 100,             # When performing the inference of alpha and bata, sequences will be selected based on this threshold if where_to_threshold is "each_time"
          para_n_jobs = 1,                    # How many cores to use
          where_to_threshold = 'each_time',   # see the explanation for Untilwhichvalue_
          parameter_ab_absmax = 30,           # the maximum of the parameter a and b
          repetition__maximum = 100,          #  the maximum of the iteration cycles to estimate theta
          threshold_NB_Poisson = 0.0001       # if lambda/r is smaller than this value, the distribution is set to Poisson.
      ):
        
    if Untilwhichvalue_ == None:
      if where_to_threshold != 'non':
        raise TypeError("Untilwhichvalue_ should be provided")
        
    if random_num==None:
      self.rng = np.random.RandomState()
    elif isinstance(random_num,int):
      self.rng = np.random.RandomState(random_num)
    elif isinstance(random_num,np.random.mtrand.RandomState):
      self.rng = random_num
    else:
      raise TypeError('ERROR: random number')
      
#    print(self.rng.random())    
    self._is_fitted = False
    self.t_rounds = t_rounds
    self.Inference_type = Inference_type
    self.theta_term_yes = theta_term_yes
    self.folder_name = folder_name
    self.save_data = save_data
    self.Regularization_ = Regularization_
    self.BiasCorrectionwith = BiasCorrectionwith
    self.Num_division = Num_division
    self.Untilwhichvalue_ = Untilwhichvalue_
    self.para_n_jobs = para_n_jobs
    self.where_to_threshold = where_to_threshold
    self.parameter_ab_absmax = parameter_ab_absmax
    self.repetition__maximum = repetition__maximum
    self.threshold_NB_Poisson = threshold_NB_Poisson

    
  """
  ######### Exp_model_fitting #############
  This function performs the first part of the two-step alrogithm (please refer to the method section of the manuscript.)
    Input data  : count data for each variant, alpha and beta
    Output data : a and b of each variant, theta and the expected value of counts.
  #########################################
  """
  def Exp_model_fitting(self,
                        data_set,             # count data
                        theta_term_saved,     # initial parameter for the theta term
                        a_neg_develop,        # alpha
                        b_neg_develop,        # beta
                        initial_parameter,    # initial parameters for a and rho0
                        nn_total              # total counts of each round
                        ):

    nn_total_mm = data_set.copy().sum().values
    Replicates_ = self.Replicates_
    t_rounds = self.t_rounds
    Regularization_ = self.Regularization_
    theta_term_yes = self.theta_term_yes
    Inference_type = self.Inference_type
    para_n_jobs = self.para_n_jobs
    parameter_ab_absmax = self.parameter_ab_absmax
    _is_fitted = self._is_fitted
    repetition__maximum = self.repetition__maximum
    threshold_NB_Poisson = self.threshold_NB_Poisson
    Initial_library_same = self.Initial_library_same
    
    
    ### store scores
    seqFreqs  = data_set.copy().values   ## count data
    scorea = np.zeros(len(data_set)*2,dtype=float).reshape((len(data_set),2)) - 100000000. ## scores (a) of each variant (first row: score, second row: 2 * standard deviation)
    f0 = np.zeros(len(data_set)*2,dtype=float).reshape((len(data_set),2)) - 100000000.  ## log(intercept) (rho0) of each variant (first row: score, second row: 2 * standard deviation)
    v_E = np.zeros(len(data_set)*4,dtype=float).reshape((len(data_set),4)) - 100000000. ## a matrix of the curvature around the maximum of loglikelihood function.
    
    if theta_term_yes == 'yes':
      repetition__ = repetition__maximum ## how many maximum repeitions to perform to determine theta term
    elif theta_term_yes == 'no':
      repetition__ = 1
      
    if _is_fitted == True:
      repetition__ = 1
    
    theta_term = theta_term_saved.copy()

    ### Below we define the likelihood function, its first and second derivatives.
    if Inference_type == 'Negbin':
      ## Negative binomial likelihood function for the exponential model
      ## input: xx_= the parameters (a, b). args = the count data. 
      ## output: log likelihood function
      def func_ab(xx_,*args):
        nn_reps = args[0]
        return_ = 0.
        for iiii in range(Replicates_):
          nn_ = nn_reps[iiii*len(t_rounds) : (iiii*len(t_rounds) + len(t_rounds))]
  #        nn_  = seqFreqs[seq,:]
          lambda_ = np.exp(xx_[0]*t_rounds + xx_[1] )*nn_total[iiii*len(t_rounds) : (iiii*len(t_rounds) + len(t_rounds))] * np.exp(theta_term) 
          rr_ = b_neg_develop*lambda_**a_neg_develop
          if min(lambda_ / rr_) > threshold_NB_Poisson:
            _dummy_ = -nbinom.logpmf(nn_,n=rr_,p=rr_/(rr_+lambda_))
          else:
            _dummy_ = -poisson.logpmf(nn_,lambda_)
          ##########            
          if Initial_library_same == "same":
            if iiii > 0:
              _dummy_[0] = 0.
          ##########            
          return_ += _dummy_.sum() + Regularization_ * xx_[0]**2
        return(return_)
      
      ## The derivative of negative binomial likelihood function
      ## input: xx_= the parameters (a, b). args = the count data. 
      ## output: first derivative of log likelihood function
      def func_ab_deri(xx_,*args):
        nn_reps = args[0]
#        nn_  = seqFreqs[seq,:]
        for iiii in range(Replicates_):
          nn_ = nn_reps[iiii*len(t_rounds) : (iiii*len(t_rounds) + len(t_rounds))]
          lambda_ = np.exp(xx_[0]*t_rounds + xx_[1])*nn_total[iiii*len(t_rounds) : (iiii*len(t_rounds) + len(t_rounds))] * np.exp(theta_term) 
          rr_ = b_neg_develop*lambda_**a_neg_develop
          if min(lambda_ / rr_) > threshold_NB_Poisson:
            aa_d = b_neg_develop * lambda_**a_neg_develop * t_rounds * (  (-1 + a_neg_develop)*(lambda_ - nn_)/(lambda_ + b_neg_develop*(lambda_)**a_neg_develop) - a_neg_develop*np.log((b_neg_develop + (lambda_)**(1-a_neg_develop))/b_neg_develop)
                  - a_neg_develop * polygamma(0,b_neg_develop*lambda_**a_neg_develop) + a_neg_develop * polygamma(0,b_neg_develop*lambda_**a_neg_develop + nn_) )
            bb_d = b_neg_develop * lambda_**a_neg_develop * (  (-1 + a_neg_develop)*(lambda_ - nn_)/(lambda_ + b_neg_develop*(lambda_)**a_neg_develop) - a_neg_develop*np.log((b_neg_develop + (lambda_)**(1-a_neg_develop))/b_neg_develop)
                  - a_neg_develop * polygamma(0,b_neg_develop*lambda_**a_neg_develop) + a_neg_develop * polygamma(0,b_neg_develop*lambda_**a_neg_develop + nn_) )
          else:
            aa_d = t_rounds*(nn_ - lambda_)
            bb_d = (nn_ - lambda_)
          if iiii == 0:
            aa_ = np.array(aa_d)
            bb_ = np.array(bb_d)
          else:
            if Initial_library_same == "same":
              aa_d[0] = 0.
              bb_d[0] = 0.
            aa_ += np.array(aa_d)
            bb_ += np.array(bb_d)
        return([-aa_.sum() + 2.*Regularization_ * xx_[0],-bb_.sum()])
      
      # The second derivative of negative binomial likelihood function
      ## input: xx_= the parameters (a, b). args = the count data. 
      ## output: second derivative of log likelihood function
      def func_ab_deri_deri(xx_,*args): ## 
        nn_reps = args[0]
#        nn_  = seqFreqs[seq,:]
        for iiii in range(Replicates_):
          nn_ = nn_reps[iiii*len(t_rounds) : (iiii*len(t_rounds) + len(t_rounds))]
          lambda_ = np.exp(xx_[0]*t_rounds + xx_[1])*nn_total[iiii*len(t_rounds) : (iiii*len(t_rounds) + len(t_rounds))] * np.exp(theta_term) 
          rr_ = b_neg_develop*lambda_**a_neg_develop
          if min(lambda_ / rr_) > threshold_NB_Poisson:
            aa_d = b_neg_develop * lambda_**a_neg_develop * t_rounds**2 * (((-1+a_neg_develop)*lambda_ * (b_neg_develop*lambda_**a_neg_develop + a_neg_develop * (2.*lambda_ + b_neg_develop*lambda_**a_neg_develop - nn_)+nn_) - a_neg_develop**2 * (lambda_ + b_neg_develop*lambda_**a_neg_develop)**2 * np.log((b_neg_develop + lambda_**(1.-a_neg_develop))/b_neg_develop))/(lambda_ + b_neg_develop*lambda_**a_neg_develop)**2  + a_neg_develop**2 * (-polygamma(0,b_neg_develop*lambda_**a_neg_develop) + polygamma(0,b_neg_develop*lambda_**a_neg_develop + nn_) - b_neg_develop * lambda_**a_neg_develop * (polygamma(1,b_neg_develop*lambda_**a_neg_develop) - polygamma(1,b_neg_develop*lambda_**a_neg_develop+nn_))))
            ab_d = b_neg_develop * lambda_**a_neg_develop * t_rounds * (((-1+a_neg_develop)*lambda_ * (b_neg_develop*lambda_**a_neg_develop + a_neg_develop * (2.*lambda_ + b_neg_develop*lambda_**a_neg_develop - nn_)+nn_) - a_neg_develop**2 * (lambda_ + b_neg_develop*lambda_**a_neg_develop)**2 * np.log((b_neg_develop + lambda_**(1.-a_neg_develop))/b_neg_develop))/(lambda_ + b_neg_develop*lambda_**a_neg_develop)**2  + a_neg_develop**2 * (-polygamma(0,b_neg_develop*lambda_**a_neg_develop) + polygamma(0,b_neg_develop*lambda_**a_neg_develop + nn_) - b_neg_develop * lambda_**a_neg_develop * (polygamma(1,b_neg_develop*lambda_**a_neg_develop) - polygamma(1,b_neg_develop*lambda_**a_neg_develop+nn_))))
            ba_d = b_neg_develop * lambda_**a_neg_develop * t_rounds * (((-1+a_neg_develop)*lambda_ * (b_neg_develop*lambda_**a_neg_develop + a_neg_develop * (2.*lambda_ + b_neg_develop*lambda_**a_neg_develop - nn_)+nn_) - a_neg_develop**2 * (lambda_ + b_neg_develop*lambda_**a_neg_develop)**2 * np.log((b_neg_develop + lambda_**(1.-a_neg_develop))/b_neg_develop))/(lambda_ + b_neg_develop*lambda_**a_neg_develop)**2  + a_neg_develop**2 * (-polygamma(0,b_neg_develop*lambda_**a_neg_develop) + polygamma(0,b_neg_develop*lambda_**a_neg_develop + nn_) - b_neg_develop * lambda_**a_neg_develop * (polygamma(1,b_neg_develop*lambda_**a_neg_develop) - polygamma(1,b_neg_develop*lambda_**a_neg_develop+nn_))))
            bb_d = b_neg_develop * lambda_**a_neg_develop * (((-1+a_neg_develop)*lambda_ * (b_neg_develop*lambda_**a_neg_develop + a_neg_develop * (2.*lambda_ + b_neg_develop*lambda_**a_neg_develop - nn_)+nn_) - a_neg_develop**2 * (lambda_ + b_neg_develop*lambda_**a_neg_develop)**2 * np.log((b_neg_develop + lambda_**(1.-a_neg_develop))/b_neg_develop))/(lambda_ + b_neg_develop*lambda_**a_neg_develop)**2   + a_neg_develop**2 * (-polygamma(0,b_neg_develop*lambda_**a_neg_develop) + polygamma(0,b_neg_develop*lambda_**a_neg_develop + nn_) - b_neg_develop * lambda_**a_neg_develop * (polygamma(1,b_neg_develop*lambda_**a_neg_develop) - polygamma(1,b_neg_develop*lambda_**a_neg_develop+nn_))))
          else:
            aa_d = (t_rounds**2) * (-lambda_)
            ab_d = - t_rounds * lambda_
            ba_d = - t_rounds * lambda_
            bb_d = -lambda_
          if iiii == 0:
            aa_ = np.array(aa_d)
            ab_ = np.array(ab_d)
            ba_ = np.array(ba_d)
            bb_ = np.array(bb_d)
          else:
            if Initial_library_same == "same":
              aa_d[0] = 0.
              ab_d[0] = 0.
              ba_d[0] = 0.
              bb_d[0] = 0.
            aa_ += np.array(aa_d)
            ab_ += np.array(ab_d)
            ba_ += np.array(ba_d)
            bb_ += np.array(bb_d)
        return(np.matrix([[-aa_.sum()+ 2.*Regularization_ ,-ab_.sum()],[-ba_.sum(),-bb_.sum()]]))

    elif Inference_type == 'Pois':
      # Poisson likelihood function
      def func_ab(xx_,*args):
        ## input: xx_= the parameters (a, b). args = the count data. 
        ## output: log likelihood function
        nn_reps = args[0]
        return_ = 0.
        for iiii in range(Replicates_):
          nn_ = nn_reps[iiii*len(t_rounds) : (iiii*len(t_rounds) + len(t_rounds))]
          lambda_ = np.exp(xx_[0]*t_rounds + xx_[1])*nn_total[iiii*len(t_rounds) : (iiii*len(t_rounds) + len(t_rounds))] * np.exp(theta_term)
          _dummy_ = -poisson.logpmf(nn_,lambda_)
          ##########            
          if Initial_library_same == "same":
            if iiii > 0:
              _dummy_[0] = 0.
          ##########            
          return_ += _dummy_.sum() + Regularization_ * xx_[0]**2
        return(return_)
    
      def func_ab_deri(xx_,*args):
        ## The first derivative of the Poisson likelihood function
        ## input: xx_= the parameters (a, b). args = the count data. 
        ## output: the first derivative of log likelihood function
        nn_reps = args[0]
        for iiii in range(Replicates_):
          nn_ = nn_reps[iiii*len(t_rounds) : (iiii*len(t_rounds) + len(t_rounds))]
          lambda_ = np.exp(xx_[0]*t_rounds + xx_[1])*nn_total[iiii*len(t_rounds) : (iiii*len(t_rounds) + len(t_rounds))] * np.exp(theta_term) 
          aa_d = t_rounds*(nn_ - lambda_)
          bb_d = (nn_ - lambda_)
          if iiii == 0:
            aa_ = np.array(aa_d)
            bb_ = np.array(bb_d)
          else:
            if Initial_library_same == "same":
              aa_d[0] = 0.
              bb_d[0] = 0.
            aa_ += np.array(aa_d)
            bb_ += np.array(bb_d)
        return([-aa_.sum()+ 2. * Regularization_ * xx_[0],-bb_.sum()])
    
      def func_ab_deri_deri(xx_,*args): 
        ## The second derivative of the Poisson likelihood function
        ## input: xx_= the parameters (a, b). args = the count data. 
        ## output: the second derivative of log likelihood function
        nn_reps = args[0]
        for iiii in range(Replicates_):
          nn_ = nn_reps[iiii*len(t_rounds) : (iiii*len(t_rounds) + len(t_rounds))]
          lambda_ = np.exp(xx_[0]*t_rounds + xx_[1])*nn_total[iiii*len(t_rounds) : (iiii*len(t_rounds) + len(t_rounds))] * np.exp(theta_term) 
          aa_d = (t_rounds**2) * (-lambda_)
          ab_d = - t_rounds * lambda_
          ba_d = - t_rounds * lambda_
          bb_d = -lambda_
          if iiii == 0:
            aa_ = np.array(aa_d)
            ab_ = np.array(ab_d)
            ba_ = np.array(ba_d)
            bb_ = np.array(bb_d)
          else:
            if Initial_library_same == "same":
              aa_d[0] = 0.
              ab_d[0] = 0.
              ba_d[0] = 0.
              bb_d[0] = 0.
            aa_ += np.array(aa_d)
            ab_ += np.array(ab_d)
            ba_ += np.array(ba_d)
            bb_ += np.array(bb_d)
        return(np.array([[-aa_.sum()+ 2. * Regularization_,-ab_.sum()],[-ba_.sum(),-bb_.sum()]]))

    
    def forparallel(seq):
      ## A function to perform the fitting
      ## input: sequence number
      ## output: a, sd of a, log(rho0), sd of log(rho0), and curveture matrix.

      initial_reference = initial_parameter[seq]
      
      ## Below, we'll first try dogleg algorithm to minimize the loglikelihood function, and if it doesn't work, we'll use Nelder-Mead method.
      try:
        ## dogleg
        test_ = minimize(func_ab, initial_reference, args=seqFreqs[seq,:], method='dogleg', jac=func_ab_deri,hess=func_ab_deri_deri)
        if abs(test_.x).max() > parameter_ab_absmax:
          suceed_ = False
        else:
          suceed_ = True
      except:
        suceed_ = False
        try:
          ___ = test_.success ## if test_ is already defined, it's ok.
        except: ## if not asign test_.success = False
          from scipy.optimize import rosen
          test_ = minimize(rosen, [1], method='Nelder-Mead', tol=1e-6)
          test_.success = False
        
      if test_.success == False or suceed_ == False:
        # Nelder-Mead
        test_ = minimize(func_ab, initial_reference, args=seqFreqs[seq,:], method='Nelder-Mead',options={'maxiter':100000},bounds=[(-parameter_ab_absmax,parameter_ab_absmax),(-parameter_ab_absmax,parameter_ab_absmax)])

      # Compute the second derivative. 
      dummy_ = func_ab_deri_deri(test_.x, seqFreqs[seq,:])
      if np.isnan(func_ab_deri_deri(test_.x, seqFreqs[seq,:]).sum()):
        print("curveture infinite")
        dummy_ = np.array([[1.000001,1],[1,1.000001]])

      # Compute Error matrix (curvature matrix)
      try:
        Error_matrix = inv(dummy_)
      except:
        Error_matrix = inv(dummy_ + np.diag([0.0001,0.0001]))
        print('Singular matrix')
        
      # This is the estimated error.
      errors_ = 2. * np.sqrt(np.diagonal(Error_matrix))
      
      # Error matrix itself.
      v_E = np.array(Error_matrix).flatten()
      
      return [test_.x[0],errors_[0],test_.x[1],errors_[1],v_E[0],v_E[1],v_E[2],v_E[3]] 
      # [a, a error, log(rho), log(rho) error, errormatrix_component1, errormatrix_component2, errormatrix_component3, errormatrix_component4]
      # Note that the error is 2* standard deviation.
      
    # A loop for computing theta.
    for theta_loop in range(repetition__): 
      # Excecuting forparallel in parallel
      dummy = Parallel(n_jobs=para_n_jobs)(delayed(forparallel)(seq) for seq in range(len(data_set)))
      scorea[:,:] = np.array(dummy)[:,0:2]
      f0[:,:] = np.array(dummy)[:,2:4]
      v_E[:,:] = np.array(dummy)[:,4:8] 
        
      ## Keep for the next initial parameters for a and log(rho)
      initial_parameter = np.vstack((scorea[:,0],f0[:,0])).transpose()

      ## Below is to compute the theta term. See the caption of the Supplementary Figure 2 for the explanation.
      if theta_term_yes == 'yes' and _is_fitted == False:
        dThetas = np.zeros(len(t_rounds))
        theta_term_old = theta_term.copy()
        for i in range(len(t_rounds)):
          theta_term[i] = - np.log(np.sum(np.exp(f0[:,0] + scorea[:,0]*t_rounds[i])))
        BB = np.polyfit(t_rounds, theta_term, 1)
        theta_term -= (BB[0]*t_rounds + BB[1])
        eps = np.sqrt(np.sum((theta_term-theta_term_old)**2)/len(theta_term))
        if np.log10(eps) < -5: # Condition to determin if the loop for computing theta converged.
          break

    ## Here is to compute lambda (expected value of counts) based on inferred parameters a and rho0
    lambda_set = np.array(data_set.values.copy(),dtype=float)
    for seq in range(len(data_set)):
      lambda_set[seq] = np.exp(scorea[seq,0]*np.tile(t_rounds,Replicates_) + f0[seq,0]+ np.tile(theta_term,Replicates_))*nn_total

    theta_term_saved = theta_term.copy()
    scorea = pd.DataFrame(scorea,index=data_set.index,columns = ["a_inf", "a_inf_err"])
    f0 = pd.DataFrame(f0,index=data_set.index,columns = ["b_inf", "b_inf_err"])
    v_E = pd.DataFrame(v_E,index=data_set.index,columns = ["v_E_1", "v_E_2","v_E_3","v_E_4"])
    lambda_set = pd.DataFrame(lambda_set,index=data_set.index,columns = data_set.columns)
    return list([scorea,f0,lambda_set,theta_term_saved,initial_parameter,v_E])
    ########################################################################################
    ####### scorea = the array of "a" and its standard deviation multipled by 2
    ####### f0 = the array of "log(rho0)" and its standard deviation multipled by 2
    ####### theta_term_saved = the theta term
    ####### initial_parameter = initial parameters for a and rho0 for the next time.
    ####### v_E = Error matrix (the curveture of the log likelihood function.)
    ########################################################################################
  
  
  """
  ######### r_fitting function #############
  This function performs the second part of the two-step alrogithm (please refer to the method section of the manuscript.)
    Input data  : count data, and its expected value (predicted by the model) for each variant, alpha and beta (This is for initial parameters.)
    Output data : a and b of each variant, theta and the expected value of counts.
  #########################################
  """
  def r_fitting_from_lambda_and_counts(self,
                                      lambda_set,        # expected value of counts 
                                      data_set,          # count data
                                      a_neg_develop,     # alpha
                                      b_neg_develop      # beta
                                      ):

    BiasCorrectionwith = self.BiasCorrectionwith
    Num_division = self.Num_division
    Initial_library_same = self.Initial_library_same
    Replicates_ = self.Replicates_
    t_rounds = self.t_rounds
    R_inference_remove = self.R_inference_remove

    ### below we define the loglikelihood function to determine r. 
    if BiasCorrectionwith == 'No': ## straigtfoward loglikelihood
      ##### loglikelihood function
      def calculateLL_min(r_lam): # calculate with a vector (and multiply minus at the end)
        L_dummy = np.sum(Ob_ * np.log(lam_1) -  Ob_ * np.log(lam_1 + r_lam) - r_lam * np.log(1.+lam_1/r_lam) + special.loggamma(Ob_ + r_lam) - special.loggamma(r_lam)  - special.loggamma(Ob_ + 1.))
        return(-L_dummy)     

      ##### first derivative ########
      def calculateLL_min_deri(r_lam): # calculate with a vector (and multiply minus at the end)
        r_ = r_lam
        lam_ = lam_1
        bb_ = np.sum((lam_ - Ob_)/(lam_ + r_) - np.log((lam_ + r_)/r_) - polygamma(0, r_) + polygamma(0, Ob_ + r_))
        return(-bb_)
      
      ##### second derivative ######## 
      def calculateLL_min_deri_deri(r_lam): # calculate with a vector (and multiply minus at the end)
        r_ = r_lam
        lam_ = lam_1
        bb_ = np.sum((lam_**2 + Ob_*r_)/(r_*(lam_ + r_)**2) - polygamma(1, r_) + polygamma(1, Ob_ + r_))
        return(-bb_)
    elif BiasCorrectionwith == 'Yes': ## correcting a bias by assuming we don't have 0 counts. 
      ##### loglikelihood #######
      def calculateLL_min(r_lam): # calculate with a vector (and multiply minus at the end)
        lam_ = lam_1
        r_ = r_lam
        L_dummy = np.sum(np.nan_to_num(Ob_ * np.log(lam_)) -  Ob_ * np.log(lam_ + r_) - r_ * np.log(1.+lam_/r_) + special.loggamma(Ob_ + r_) - special.loggamma(r_)  - special.loggamma(Ob_ + 1.))
        L_dummy -= np.log(1-(1 + lam_/r_)**(-r_)) * len(Ob_)
        return(-L_dummy)     
        
      ##### first derivative ########
      def calculateLL_min_deri(r_lam): # calculate with a vector (and multiply minus at the end)
        lam_ = lam_1
        r_ = r_lam
        bb_ = np.sum((Ob_ - ((lam_ + r_)/r_)**r_ * (-lam_ + Ob_ + np.log((lam_ + r_)/r_) * (lam_ + r_)) - (lam_ + r_) * (-1 + ((lam_ + r_)/r_)**r_)* (polygamma(0, r_) - polygamma(0, Ob_ + r_)))/((lam_ + r_)* (-1 + ((lam_ + r_)/r_)**r_)))
        return(-bb_)
  
      ##### second derivative ######## 
      def calculateLL_min_deri_deri(r_lam): # calculate with a vector (and multiply minus at the end)
        lam_ = lam_1
        r_ = r_lam
        bb_ = np.sum(
          (Ob_ * r_ + ((lam_ + r_)/r_)**(2* r_) * (lam_**2 + Ob_ *r_) + ((lam_ + r_)/r_)**r_ * (lam_**2 * (-1 + r_) - 2 * Ob_ * r_ + np.log((lam_ + r_)/r_) * r_ * (lam_ + r_)* (-2 * lam_ + np.log((lam_ + r_)/r_) * (lam_ + r_))) - r_ * (lam_ + r_)**2 * (-1 + ((lam_ + r_)/r_)**r_)**2 * (polygamma(1, r_) - polygamma(1, Ob_ + r_)))/(r_ * (lam_ + r_)**2 * (-1 + ((lam_ + r_)/r_)**r_)**2)
          )
        return(-bb_)
    elif BiasCorrectionwith == 'Yes_inverse': ## correcting a bias by assuming we don't have 0 counts. We also compute directly 1/r instead of r.
      ##### log likelihood #######
      def calculateLL_min(r_lam): # calculate with a vector (and multiply minus at the end)
        lam_ = lam_1
        r_ = 1./r_lam
        L_dummy = np.sum(np.nan_to_num(Ob_ * np.log(lam_)) -  Ob_ * np.log(lam_ + r_) - r_ * np.log(1.+lam_/r_) + special.loggamma(Ob_ + r_) - special.loggamma(r_)  - special.loggamma(Ob_ + 1.))
        L_dummy -= np.log(1-(1 + lam_/r_)**(-r_)) * len(Ob_)
        return(-L_dummy)     
        
      ##### first derivative ########
      def calculateLL_min_deri(r_lam): # calculate with a vector (and multiply minus at the end)
        lam_ = lam_1
        r_ = 1./r_lam
        bb_ = np.sum(r_**2  * ((np.log(1 + lam_/r_) * (1 + lam_/r_) + (-lam_ + Ob_)/r_)/(1 - (1 + lam_/r_)**(1 - r_) + lam_/r_) - Ob_/((-1 + (1 + lam_/r_)**r_) * (1 + lam_/r_) * r_) + polygamma(0, r_) - polygamma(0, Ob_ + r_)))
        return(-bb_)
  
      ##### second derivative ######## 
      def calculateLL_min_deri_deri(r_lam): # calculate with a vector (and multiply minus at the end)
        lam_ = lam_1
        r_ = 1./r_lam
        bb_ = np.sum(
          (r_**4 * ((np.log(1 + lam_/r_) * (1 + lam_/r_) * (np.log(1 + lam_/r_) * (1 + lam_/r_) + (2 * (1 + lam_ * (-1 + 1/r_)))/r_) + (2 * Ob_ + lam_ * (-2 + lam_ - (3 * lam_)/r_ + (4 * Ob_)/r_))/r_**2) * (1 + lam_/r_)**r_ - (Ob_ * (1 + (2 * lam_)/r_))/ r_**2 
          - ((1 + lam_/r_)**(2 * r_)*(2 * np.log(1 + lam_/r_) * (1 + lam_/r_)**2 + (Ob_ + lam_ * (-2 - (3 * lam_)/r_ + (2 * Ob_)/r_))/r_))/r_ - (2 * (-1 + (1 + lam_/r_)**r_)**2 * (1 + lam_/r_)**2 * polygamma(0, r_))/r_ + (2 * (-1 + (1 + lam_/r_)**r_)**2 * (1 +
          lam_/r_)**2 * polygamma(0, Ob_ + r_))/r_ - polygamma(1,  r_) + (2 * (1 + lam_/r_)**(2 + r_) - (1 + lam_/r_)**(2 + 2 * r_) - (lam_ * (2 + lam_/r_))/r_) * (polygamma(1, r_) - polygamma(1, Ob_ + r_)) + polygamma(1, Ob_ + r_)))/((-1 + (1 + lam_/r_)**r_)**2 * (1 + lam_/r_)**2)
          )
        return(-bb_)
    else:
      raise TypeError('ERROR: BiasCorrectionwith')
  
    ## below is for the case with replicates with shared initial library. 
    if R_inference_remove == None: ## 
      if Initial_library_same == "same": ## When the library shares the same library, the inference of r is biased. To avoid it, we set R_inference_remove to be the round we want to remove.
        lambda_set = np.delete(lambda_set,np.arange(0,lambda_set.shape[1],len(t_rounds)),axis=1)
        data_set = np.delete(data_set,np.arange(0,data_set.shape[1],len(t_rounds)),axis=1)
    elif Initial_library_same == None: ## Below is just to remove a library based on R_inference_remove. This is for a test. Don't use it in a standard application. 
      lambda_set = np.delete(lambda_set,np.arange(0+R_inference_remove,R_inference_remove + lambda_set.shape[1],len(t_rounds)),axis=1)
      data_set = np.delete(data_set,np.arange(0+R_inference_remove,R_inference_remove + data_set.shape[1],len(t_rounds)),axis=1)
      lambda_set = lambda_set[:,self.where_to_take_lambda]
      data_set = data_set[:,self.where_to_take_lambda]

    ## First flatten the lambda (expected value of counts)
    dummy_lambda_determine = lambda_set.flatten().copy()
    ## We'll divide the lambda space into "Num_division" bins using percentile: 
    lambda_domain = np.percentile(dummy_lambda_determine,np.linspace(10,90,Num_division))
    ## We don't use the domain whose lambda value is less than 1, because count data are less informative in that regime.
    wheretodo = np.where((lambda_domain>1))[0]
    
    minresult_set = list(np.arange(len(wheretodo)-1)) # We'll store a result of fitting here.
    r_obtained = np.zeros((len(wheretodo)-1)) # We'll store the obtained r value here.
    for i in range(len(wheretodo)-1): # loop for each bin
      # select the index whose lambda is in the specified bin.
      dummy_posi = np.where((lambda_set.flatten() >= lambda_domain[wheretodo[i]]) & (lambda_set.flatten() < lambda_domain[wheretodo[i+1]]))[0]
      # Representative value of lambda
      lam_1 = (lambda_domain[wheretodo[i]] + lambda_domain[wheretodo[i+1]])/2
      # The counts in this bin.
      Ob_ = data_set.flatten()[dummy_posi]

      # If we use bias correction likelihood funciton without 0 counts (see above), we remove 0 counts:
      if BiasCorrectionwith == 'Yes' or BiasCorrectionwith=='Yes_inverse':
        Ob_ = np.array(Ob_[Ob_>0])
        
      # Inferring r. If we use BiasCorrectionwith=='Yes_inverse', the inverse of r will be estimated. 
      if BiasCorrectionwith=='Yes_inverse': 
        minresult_set[i] = minimize(calculateLL_min, x0= [1/(b_neg_develop[0]*lam_1**a_neg_develop[0])], method='L-BFGS-B',bounds=[(1/(100*lam_1),1000)], jac=calculateLL_min_deri)#,hess=calculateLL_min_deri_deri)
      else:
        minresult_set[i] = minimize(calculateLL_min, x0= [b_neg_develop[0]*lam_1**a_neg_develop[0]], method='L-BFGS-B',bounds=[(1/1000,100*lam_1)], jac=calculateLL_min_deri)#,hess=calculateLL_min_deri_deri)
  
      # In case of the inference of 1/r, we have to compute the inverse to get r.
      if BiasCorrectionwith=='Yes_inverse':
        r_obtained[i] = 1./minresult_set[i].x[0]
      else:
        r_obtained[i] = minresult_set[i].x[0]
    
    biased_r_obtained = r_obtained.copy() ## Obtained r
    x_dum = (lambda_domain[wheretodo][:-1] +  lambda_domain[wheretodo][1:])/2 ## Corresponding lambda value
    y_dum = np.log(r_obtained[:]) ## logarithm of r
    where_nonnan = ~((np.isnan(x_dum)) | (np.isnan(y_dum))) # look for the domain on which no nan values appear.
  
    # Below is to fit a linear function to the obtained r(lambda), from which alpha and beta is determined.
    _dummy_ = np.polyfit(np.log(x_dum[where_nonnan]),y_dum[where_nonnan],1) # This is for a futting function for r = beta lambda**alpha

    ## Below is to implement a threshold for the alpha and beta for the fitting function:
    ######################## slope fixed
    ######################## slope fixed
    fixed_slope = 0.01 ## The lower bound of alpha
    fixed_bb = 0.1 ## The lower bound of beta
    fixed_slope_2 = 1.5 ## The upper bound of alpha
    xx_dfit = np.log(x_dum[where_nonnan])
    yy_dfit = y_dum[where_nonnan]
    if (_dummy_[0] > fixed_slope_2 or _dummy_[0] < fixed_slope) and np.exp(_dummy_[1]) < fixed_bb:
      if _dummy_[0] > fixed_slope_2:
        fixed_slope_dummy = fixed_slope_2
      elif _dummy_[0] < fixed_slope:
        fixed_slope_dummy = fixed_slope
      a_neg_develop[:] = fixed_slope_dummy
      b_neg_develop[:] = fixed_bb
    elif _dummy_[0] > fixed_slope_2 or _dummy_[0] < fixed_slope:
      if _dummy_[0] > fixed_slope_2:
        fixed_slope_dummy = fixed_slope_2
      elif _dummy_[0] < fixed_slope:
        fixed_slope_dummy = fixed_slope
      a_neg_develop[:] = fixed_slope_dummy
      b_neg_develop[:] = max([np.exp(np.mean(yy_dfit-fixed_slope_dummy*xx_dfit)),fixed_bb])
    elif np.exp(_dummy_[1]) < fixed_bb:
      b_neg_develop[:] = fixed_bb
      y_data_dummy = yy_dfit - np.log(fixed_bb)
      slope_determined = np.sum(y_data_dummy*xx_dfit)/np.sum(xx_dfit**2)
      a_neg_develop[:] = max([slope_determined,fixed_slope])
      a_neg_develop[:] = min([a_neg_develop.mean(),fixed_slope_2])
    else:
      a_neg_develop[:] = _dummy_[0]
      b_neg_develop[:] = np.exp(_dummy_[1])

    ## This is fiting a polynomial function to the r(lambda), which is used later to unbias the alpha-beta estiamtes.
    _dummy_ = np.polyfit(np.log(x_dum[where_nonnan]),y_dum[where_nonnan],6)
    dummy_func = np.poly1d(_dummy_)
    return list([x_dum,biased_r_obtained,a_neg_develop,b_neg_develop,dummy_func])
    ########################################################################################
    ### x_dum = lambda value corresponding to the obtained r
    ### biased_r_obtained = obtained r
    ### a_neg_develop = obtained alpha
    ### b_neg_develop = obtained beta
    ### dummy_func = Fitting funciton to r, used later for unbiasing.
    ########################################################################################


  """
  ######### synthetic data generator #############
  This function generates synthetic data based on lambda (expected value of counts), alpha, beta.
    Input data  : lambda (expected value of counts), alpha, beta
    Output data : count data generated synthetically
  ################################################
  """
  def making_synthetic_data(self,
                            lambda_set, # lambda
                            a_neg_develop, # alpha
                            b_neg_develop # beta
                            ):
    """
     lambda_set = self.lambda_set.copy()
     a_neg_develop = self.a_neg_develop_synthesis.copy()
     b_neg_develop = self.b_neg_develop_synthesis.copy()
    """
    rng = self.rng
    # compute r
    rr_synthesis = b_neg_develop.mean() * (lambda_set.values)**a_neg_develop.mean()
    # generate synthetic data using negaative binomial distribution
    data_set_synthesis = pd.DataFrame(rng.negative_binomial(n=rr_synthesis,p=rr_synthesis/(rr_synthesis+lambda_set.values)),index=lambda_set.index,columns=lambda_set.columns)
    return data_set_synthesis
    ########################################################################################
    ### data_set_synthesis = synthetic data counts
    ########################################################################################



  """
  ######### trimming the count data using a threshold ##########
  This function trims count data using a pre-determined threshold.
    Input data  : count data
    Output data : trimmed count data
  #############################################################
  """
  def thresholding_data(self,data_set):
    data_set_th = data_set.copy()
    Untilwhichvalue_ = self.Untilwhichvalue_
    Replicates_ = self.Replicates_
    
    if Untilwhichvalue_ == "PRODUCT":
      data_set_th = data_set_th[data_set_th.prod(axis=1)>0]
    else:
      ## Using a pre-determined threshold (Untilwhichvalue_), trim the data
      data_set_th = data_set_th[data_set_th.sum(axis=1) > int(Untilwhichvalue_*Replicates_)]
      # In addition to this, we also remove the sequences that only appear twice.
      if len(self.t_rounds) > 2:
        data_set_th = data_set_th[data_set_th.replace(0, np.nan).count(axis=1) > int(2*Replicates_)]
      else:
        pass
    return data_set_th
    ########################################################################################
    ### data_set_th = Trimmed data
    ########################################################################################


  """
  ######### Preparing the initial parameters for the inference of a and log(rho0) ##########
  The initial parameter of a is determined as (count[last time] - count[initial time]) / (last time - initial time).
  log(rho0) is determined from "a" as an intercept. 
  If replicates are avialble, average it over replicate. 
    Input data  : count data, the total NGS reads for each round, time point data
    Output data : initial parameters for a and log(rho0)
  #########################################################################################
  """
  def initial_condition_for_aa_bb(self,
                                  data_set, ## count data
                                  nn_total, ## the total count for each round
                                  t_rounds  ## time point data (e.g., [0,1,2,3,4,5] with 5 selection rounds.)
                                  ):
    Replicates_ = self.Replicates_
    dummy__inital = np.log((data_set+1)/nn_total)
    for ii in range(Replicates_):
      if ii == 0:
        # estiamte for a
        aa_initial = (dummy__inital[:,ii*len(t_rounds) + len(t_rounds)-1] -dummy__inital[:,ii*len(t_rounds)])/(t_rounds[-1] - t_rounds[0])
        # estimate for log(rho0). "b" is log(rho0)
        bb_initial = dummy__inital[:,ii*len(t_rounds)] - aa_initial*t_rounds[0]
      else:
        aa_initial += (dummy__inital[:,ii*len(t_rounds) + len(t_rounds)-1] -dummy__inital[:,ii*len(t_rounds)])/(t_rounds[-1] - t_rounds[0])
        bb_initial += dummy__inital[:,ii*len(t_rounds)] - aa_initial*t_rounds[0]
    aa_initial /= Replicates_
    bb_initial /= Replicates_
    return np.vstack((aa_initial,bb_initial)).transpose()
    ########################################################################################
    ### np.vstack((aa_initial,bb_initial)).transpose() = Initial parameters for a and rho0
    ########################################################################################
  
    
  """
  ######### 
  a function to infer alpha, beta, "a" and log(rho0) from count data  ############
    Input data  : count data
    Output data : self 
  #########################################################################################
  """
  def fit(self,
          _data_set_,                  # count data (pandas DataFrame. row: sequences, columns: different times). It's recommended to use the data that are already pre-thresholded using the counts.
          Replicates_=1,               # the number of replicates
          Initial_library_same = None, # if there are replicates, this has to be provided: Either None or "same".  
          negbin_iterate = 30,         # how many iterations for the two-step algorithm there are. (see the manuscript for more details)
          Fixed_abneg=None,            # If None, we will esimate alpha and beta together with a and log(rho0). If an array of size 2 is provided, alpha and beta are fixed to these values, and we will infer only a and log(rho0) 
          R_inference_remove = None,   # This is if we want to remove a round when inferring r. It's not necessary for a standard use. 
          ini_abneg=np.array([0.1,1.]) # The initial parameters for alpha (a_neg_develop) and beta (b_neg_develop).
          ):
    print("fit START!")

    # raising error if the parameters are not compatible
    if R_inference_remove != None:
      if Initial_library_same != None:
        raise TypeError("R_inference_remove should be non None only if Initial_library_same is None.")
    
    # raising error if the parameters are not compatible
    if Initial_library_same == None:
      if Replicates_ > 1:
        if abs(_data_set_.iloc[:,0] - _data_set_.iloc[:,len(self.t_rounds)]).sum() < 1e-8:
          raise TypeError('Initial library is common. Set Initial_library_same to be "same".')          

    if not isinstance(_data_set_, pd.DataFrame):
      raise TypeError('Use pandas dataframe for the input data')
    self.data_set = _data_set_.copy()
    self.data_set_all = _data_set_.copy()
    self.Replicates_ = Replicates_
    self.Initial_library_same = Initial_library_same
    self.Fixed_abneg = Fixed_abneg
    self.R_inference_remove = R_inference_remove

    self.a_neg_develop = np.repeat(ini_abneg[0],len(self.t_rounds)) # setting initial parameters (alpha)
    self.b_neg_develop = np.repeat(ini_abneg[1],len(self.t_rounds)) # setting initial parameters (beta)
    
    ## if Fixed_abneg is provided, we set self.a_neg_develop and self.b_neg_develop to be those values.
    if Fixed_abneg != None:
      if negbin_iterate != 1:
        raise TypeError("negbin_iterate has to be set 1.")
      if len(Fixed_abneg) != 2:
        raise TypeError("Fixed_abneg has to be an array of size 2.")
      self.a_neg_develop = np.repeat(Fixed_abneg[0],len(self.t_rounds))
      self.b_neg_develop = np.repeat(Fixed_abneg[1],len(self.t_rounds))

    ## trimming the data ################
    if self.where_to_threshold == 'non': # not using trimming.
      self.nn_total = self.data_set.sum(axis=0).values
    elif self.where_to_threshold == 'each_time': ## it uses a scaling for theta to reduce the computational cost. (Please refer to the method section of the manuscript.)
      ## theta(t) = - log[sum(e(a_i t + b_i)) * Ntot(t) / sum(n_i(t))], where sum is taken over only for thresholded data. (2022/6/28)
      self.nn_total = self.data_set.sum(axis=0).values
      self.data_set = self.thresholding_data(self.data_set)
    else:
      raise TypeError("where_to_threshold should be either non, or each_time")

    ## raising error if the parameters are not compatible
    if len(self.t_rounds)*Replicates_ != self.data_set.shape[1]:
      raise TypeError("t_rounds are not compatible with data_set")
    
    ## set the initial parameters for a and log(rho0). See the funciton "initial_condition_for_aa_bb". 
    self.initial_parameter = self.initial_condition_for_aa_bb(self.data_set.values.copy(),
                                                         self.nn_total.copy(),
                                                         self.t_rounds.copy())

    self.firsttime_ = 'yes'
    if self.Inference_type == 'Pois': # In case of Poisson inference, the two-step method is not necesarry. Consequently, it sets the number of loop to be 1.
      ## The number of the loops for the two-step method.
      self.repetition_num_negbin = 1
    elif self.Inference_type == 'Negbin':
      ## The number of the loops for the two-step method.
      self.repetition_num_negbin = negbin_iterate

    ## loop for the two-step method.
    for rep_NB_i in range(self.repetition_num_negbin):
      print(rep_NB_i)
      if rep_NB_i == 0:
        ## define the initial parameter for theta.
        self.theta_term_saved = np.array(self.t_rounds)*0.


      ### apply "Exp_model_fitting" function to obtain
      # self.scorea            : a    
      # self.f0                 : log(rho0)
      # self.lambda_set         : lambda (expected counts)
      # self.theta_term_saved   : theta term  
      # self.initial_parameter  : the next initial parameters for a and log(rho0)
      # self.v_E                : the error matrix. 
      # See "Exp_model_fitting" for more details.
      self.scorea, self.f0, self.lambda_set, self.theta_term_saved, self.initial_parameter, self.v_E = self.Exp_model_fitting(
                                                                                                        self.data_set.copy(),
                                                                                                        self.theta_term_saved.copy(),
                                                                                                        self.a_neg_develop.copy(),
                                                                                                        self.b_neg_develop.copy(),
                                                                                                        self.initial_parameter.copy(),
                                                                                                        self.nn_total.copy())


      if self.Inference_type == 'Negbin' and Fixed_abneg==None:
        ## only first time: alpha (self.a_neg_develop) and beta (self.b_neg_develop) need to be estimated.
        if self.firsttime_ == 'yes':
          _, _, self.a_neg_develop, self.b_neg_develop, _ = self.r_fitting_from_lambda_and_counts(
                                                                                          self.lambda_set.values.copy(),
                                                                                          self.data_set.values.copy(),
                                                                                          self.a_neg_develop.copy(),
                                                                                          self.b_neg_develop.copy())  
        ## By using the estiamted alpha and beta, below an unbiasing procedure is performed. See Fig.S2 for more details.
        self.a_neg_develop_synthesis = self.a_neg_develop.copy()
        self.b_neg_develop_synthesis = self.b_neg_develop.copy()

        ### generate synthetic data based on lambda (expected value for count), alpha and beta.
        self.data_set_synthesis = self.making_synthetic_data(
                                                   self.lambda_set.copy(),
                                                   self.a_neg_develop_synthesis.copy(),
                                                   self.b_neg_develop_synthesis.copy())

        ### If replicates are available, and if they share the initial library, make the generated synthetic data have this condition.
        if Initial_library_same == "same":
          for ii_rep in range(Replicates_):
            if ii_rep != 0:
              self.data_set_synthesis.iloc[:,ii_rep*len(self.t_rounds)] = self.data_set_synthesis.iloc[:,0]

        ### if trimming condition is "each_time", we also perform trimming here for the synthetic data.
        if self.where_to_threshold == 'each_time':
          self.data_set_synthesis = self.thresholding_data(self.data_set_synthesis.copy())
          
        ### determine the initial parameters for a and rho0.
        self.initial_parameter_synthetic = self.initial_condition_for_aa_bb(
                                                                       self.data_set_synthesis.values.copy(),
                                                                       self.nn_total.copy(),
                                                                       self.t_rounds.copy())  
                                                                       
        
        ### initial setting of theta term
        self.theta_term_saved_synthetic = np.array(self.t_rounds)*0.

        ### below is to determine the total reads for each round. This changes depending on trimming condition. For "each_time", we will use the original values. 
        self.nn_total_synthetic = self.data_set_synthesis.sum(axis=0).values
        if self.where_to_threshold == 'non':
          dummy_nn_total_synthetic = self.nn_total_synthetic.copy()
        elif self.where_to_threshold == 'each_time':
          dummy_nn_total_synthetic = self.nn_total.copy()


        ### apply "Exp_model_fitting" function to synthetic data to obtain
        # self.scorea_synthesis            : a for synthetic data    
        # self.f0_synthesis                 : log(rho0) for synthetic data
        # self.lambda_set_synthesis         : lambda (expected counts) for synthetic data
        # self.theta_term_saved_synthetic   : theta term for synthetic data
        # self.v_E_synthesis                : the error matrix for synthetic data
        # See "Exp_model_fitting" for more details.
        self.scorea_synthesis, self.f0_synthesis, self.lambda_set_synthesis, self.theta_term_saved_synthetic, _, self.v_E_synthesis = self.Exp_model_fitting(
                                                                                                                                      self.data_set_synthesis.copy(),
                                                                                                                                      self.theta_term_saved_synthetic.copy(),
                                                                                                                                      self.a_neg_develop_synthesis.copy(),
                                                                                                                                      self.b_neg_develop_synthesis.copy(),
                                                                                                                                      self.initial_parameter_synthetic.copy(),
                                                                                                                                      dummy_nn_total_synthetic.copy())                                                                       
        
        
        
        ### apply "r_fitting_from_lambda_and_counts" function to synthetic data to obtain 
        # self.x_dum_synthesis        : values of lambda 
        # self.r_obtained_synthesis   : values of r (synthetic data) corresponding to self.x_dum_synthesis
        # self.dummy_func             : fitting function of r(lambda)
        # See "r_fitting_from_lambda_and_counts" for more details.
        self.x_dum_synthesis,self.r_obtained_synthesis, _, _, self.dummy_func = self.r_fitting_from_lambda_and_counts(
                                                                              self.lambda_set_synthesis.values.copy(),
                                                                              self.data_set_synthesis.values.copy(),
                                                                              self.a_neg_develop_synthesis.copy(),
                                                                              self.b_neg_develop_synthesis.copy())    
                                                                              
        ### apply "r_fitting_from_lambda_and_counts" function to real data to obtain 
        # self.x_dum        : values of lambda 
        # self.r_obtained   : values of r corresponding to self.x_dum_synthesis
        # See "r_fitting_from_lambda_and_counts" for more details.
        self.x_dum, self.r_obtained, _, _, _ = self.r_fitting_from_lambda_and_counts(
                                                                                  self.lambda_set.values.copy(),
                                                                                  self.data_set.values.copy(),
                                                                                  self.a_neg_develop.copy(),
                                                                                  self.b_neg_develop.copy())  

        ## To use self.dummy_func outside of the defined region, "dummy_func_"" is defined. 
        where_nonnan_syn = ~((np.isnan(self.x_dum_synthesis)) | (np.isnan(np.log(self.r_obtained_synthesis))))
        def dummy_func_(xxxxx):
          return_value = self.dummy_func(xxxxx)
          return_value[xxxxx < min(np.log(self.x_dum_synthesis[where_nonnan_syn]))] = self.dummy_func(min(np.log(self.x_dum_synthesis[where_nonnan_syn]))) # take a constant value outside
          return_value[xxxxx > max(np.log(self.x_dum_synthesis[where_nonnan_syn]))] = self.dummy_func(max(np.log(self.x_dum_synthesis[where_nonnan_syn]))) # take a constant value outside
          return(return_value)


        ## Here, using the fitting function obtaeind from synthetic data, we unbias self.r_obtained ####### 
        self.bias_ = np.exp(dummy_func_(np.log(self.x_dum)))  / (self.b_neg_develop_synthesis[0]*self.x_dum**self.a_neg_develop_synthesis[0])
        self.unbiased_r_obtained=self.r_obtained/self.bias_
        self.y_dum = np.log(self.unbiased_r_obtained[:])
        where_nonnan = ~((np.isnan(self.x_dum)) | (np.isnan(self.y_dum)))
        _dummy_ = np.polyfit(np.log(self.x_dum[where_nonnan]),self.y_dum[where_nonnan],1) # r(lambda) = beta lambda**alpha

        ###############################################################################
        ## Fitting a linear line to the estimated unbiased r, we obtain alpha and beta.
        ######################## slope fixed ##########################################
        fixed_slope = 0.01 # lower bound of alpha
        fixed_bb = 0.1 # lower bound of beta
        fixed_slope_2 = 1.5 # upper bound of alpha
        ## below is to apply these bounds on estimated alpha and beta.
        xx_dfit = np.log(self.x_dum[where_nonnan])
        yy_dfit = self.y_dum[where_nonnan]
        if (_dummy_[0] > fixed_slope_2 or _dummy_[0] < fixed_slope) and np.exp(_dummy_[1]) < fixed_bb:
          if _dummy_[0] > fixed_slope_2:
            fixed_slope_dummy = fixed_slope_2
          elif _dummy_[0] < fixed_slope:
            fixed_slope_dummy = fixed_slope
          self.a_neg_develop[:] += fixed_slope_dummy
          self.b_neg_develop[:] += fixed_bb
        elif _dummy_[0] > fixed_slope_2 or _dummy_[0] < fixed_slope:
          if _dummy_[0] > fixed_slope_2:
            fixed_slope_dummy = fixed_slope_2
          elif _dummy_[0] < fixed_slope:
            fixed_slope_dummy = fixed_slope
          self.a_neg_develop[:] += fixed_slope_dummy
          self.b_neg_develop[:] += max([np.exp(np.mean(yy_dfit-fixed_slope_dummy*xx_dfit)),fixed_bb])
        elif np.exp(_dummy_[1]) < fixed_bb:
          self.b_neg_develop[:] += fixed_bb
          y_data_dummy = yy_dfit - np.log(fixed_bb)
          slope_determined = np.sum(y_data_dummy*xx_dfit)/np.sum(xx_dfit**2)
          slope_determined_dummy = max([slope_determined,fixed_slope])
          slope_determined_dummy = min([slope_determined_dummy,fixed_slope_2])
          self.a_neg_develop[:] += slope_determined_dummy
        else:
          self.a_neg_develop[:] += _dummy_[0]
          self.b_neg_develop[:] += np.exp(_dummy_[1])
        self.a_neg_develop /= 2.
        self.b_neg_develop /= 2.
        print(self.a_neg_develop[0],self.b_neg_develop[0])
        self.firsttime_ = 'no'
      
      ## alpha and beta are stored for all the iteration step.  
      if rep_NB_i == 0:
        self.a_neg_develop_all = self.a_neg_develop.copy()
        self.b_neg_develop_all = self.b_neg_develop.copy()
        self.theta_term_saved_all = self.theta_term_saved.copy()
      else:
        self.a_neg_develop_all = np.vstack((self.a_neg_develop_all,self.a_neg_develop.copy()))
        self.b_neg_develop_all = np.vstack((self.b_neg_develop_all,self.b_neg_develop.copy()))
        self.theta_term_saved_all = np.vstack((self.theta_term_saved_all,self.theta_term_saved.copy()))
        
      ## save the data
      if self.save_data == 'yes':
        with open(self.folder_name + '/save_'+str(rep_NB_i)+'.pkl', 'wb') as outp:
          pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    print("fit DONE!")
    self._is_fitted = True
    return self
    ########################################################################################
    ### self
    ########################################################################################

  
  
  
  """
  ######### 
  After determining alpha and beta with theta, we detemine a and log(rho0) for all the sequences, including those exluded (because of Untilwhichvalue_). ##########
  This function can be available only after using "fit" function. 
    Output data : self
  #########################################################################################
  """
  def fit_after_fixing_parameters(self,
                                  LowerBound = 0,         ## If you want to trim the data, use this parameter. (Normally set LowerBound=0)
                                  average_range_0 = 10,   ## Average length: alpha and beta are averaged over the last "average_range_0" iterations of two-step algorithms.  
                                  average_range_1 = None  ## Keep average_range_1 = None.
                                  ):
    print("fit_after_fixing_parameters START!")
    if self.Fixed_abneg != None:
      if average_range_0 != 1:
        raise TypeError("average_range_0 needs to be 1.")
    if average_range_1 == None:
      average_range_1_dum = self.repetition_num_negbin
    else:
      average_range_1_dum = average_range_1
    average_range_0_dum = average_range_1_dum - average_range_0

    _data_set_ = self.data_set_all.copy()
    _data_set_ = _data_set_[_data_set_.sum(axis=1)>LowerBound]
    if self._is_fitted == False:
      raise TypeError("NotFitted_Yet")
    if self.Inference_type == 'Negbin': ### Taking the average of alpha and beta.
      a_neg_develop = self.a_neg_develop_all[average_range_0_dum:average_range_1_dum].mean(axis=0).copy()
      b_neg_develop = self.b_neg_develop_all[average_range_0_dum:average_range_1_dum].mean(axis=0).copy()
      theta_term_saved = self.theta_term_saved_all[average_range_0_dum:average_range_1_dum].mean(axis=0).copy()
    elif self.Inference_type == 'Pois': ### In case of Poisson, these parameters won't be used anyway.
      a_neg_develop = self.a_neg_develop_all.copy()
      b_neg_develop = self.b_neg_develop_all.copy()
      theta_term_saved = self.theta_term_saved_all.copy()
    else:
      raise TypeError("We don't have that option for Inference_type.")
    nn_total = self.nn_total.copy()
    self.nn_total_all = nn_total.copy()

    ### Compute the initial parameters for a and log(rho0)
    initial_parameter = self.initial_condition_for_aa_bb(_data_set_.values.copy(),
                                                         nn_total.copy(),
                                                         self.t_rounds.copy())
                                                              
    ### apply "Exp_model_fitting" function to obtain
    # self.scorea_all            : a for all variants
    # self.f0_all                 : log(rho0) for all variants
    # self.lambda_set_all         : lambda (expected counts) for all variants
    # self.v_E_all                : the error matrix for all variants
    # See "Exp_model_fitting" for more details.
    self.scorea_all, self.f0_all, self.lambda_set_all, _, _, self.v_E_all = self.Exp_model_fitting(_data_set_.copy(),
                                                                                                      theta_term_saved.copy(),
                                                                                                      a_neg_develop.copy(),
                                                                                                      b_neg_develop.copy(),
                                                                                                      initial_parameter.copy(),
                                                                                                      nn_total.copy())
    ### save the self
    if self.save_data == 'yes':
      with open(self.folder_name + '/save_all.pkl', 'wb') as outp:
        pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
    print("fit_after_fixing_parameters END!")
    return self
    ########################################################################################
    ### self
    ########################################################################################
    



  """
  ######### For two-time data set with replicates. #######################################################
  This function is to estimate lambda when the data has only two time points. In this case, we don't need "Exp_model_fitting".
    Output: lambda (expected value of counts). This is estimated as the average count over all the replicates.
  #########################################################################################
  """
  def lambda_when_two_time_dataset(self,data_set):
    t_rounds = np.array(self.t_rounds)
    for _p_ in [0,1]: ## _p_=0 means the initial round, _p_=1 means the second round.
      
      ## first, estimate the average of frequency (count/total count). 
      first_ = 0
      for irep in data_set.columns[np.arange(_p_+0,data_set.shape[1],len(t_rounds))]:
        if first_ == 0:
          _dummy_ = data_set[irep]/data_set[irep].sum()
          first_ += 1
        else:
          _dummy_ += data_set[irep]/data_set[irep].sum()
          first_ += 1
      _dummy_ /= first_
      
      ## second, based on the estimated average frequency, compute the average count.
      first_ = 0
      for irep in data_set.columns[np.arange(_p_+0,data_set.shape[1],len(t_rounds))]:
        if first_ == 0:
          _dummy_2 = _dummy_*data_set[irep].sum()
          first_ += 1
        else:
          _dummy_2 = pd.concat((_dummy_2, _dummy_*data_set[irep].sum()),axis=1)
          
      ## change the column name.
      _dummy_2.columns = data_set.columns[np.arange(_p_+0, data_set.shape[1],len(t_rounds))]
      if _p_ == 0:
        _dummy_all = _dummy_2.copy()
      else:
        _dummy_all = pd.concat((_dummy_all,_dummy_2.copy()),axis=1) # combine lambda for _p_=0 and 1.
    ## change the order of columns    
    _dummy_all = _dummy_all.loc[:,data_set.columns] 
    return _dummy_all
    ########################################################################################
    ### _dummy_all = lambda (expected value of counts)
    ########################################################################################
    

  """
  ######### For two-time data set with replicates. #######################################################
  This function is to calculate the value of r from a given value of lambda with a fitting function.
    Input: lambda, fitting function
    Output: r
  ########################################################################################################
  """
  def rr_removeoutside(self,
                       xx, ## value of lambda
                       x_min, ## this is not used.
                       x_max, ## this is not used.
                       _Fit_func_ ## a fitting function.
                       ):
    return_value = np.exp(_Fit_func_(np.log(xx)))
    return(return_value)
    ########################################################################################
    ### return_value = r
    ########################################################################################


  """
  ######### For two-time data set with replicates. #######################################################
  This function is to infer alpha, beta, "a" and log(rho0) from count data 
    Input data  : count data
    Output data : obtained alpha and beta for t=1
                  obtained alpha and beta for t=0
                  lambda values (the expected value of counts)
  ########################################################################################################
  """
  def fit_for_two_time_dataset(self,
          _data_set_,                  # count data (pandas DataFrame. row: sequences, columns: different times). When replicate data is available, pile up (Initial-rep1, Final-rep1, Initial-rep2, Final-rep2,...)
          Replicates_,                 # the number of replicates
          Initial_library_same,        # if there are replicates, it has to be provided. either None or "same".  
          negbin_iterate = 30,         # how many iterations for the two-step algorithm there are. (see the manuscript for more details)
          average_length = 10,         # how many iterations to take the average of alpha and beta
          ini_abneg=np.array([1.,1.])  # The initial parameters for alpha (a_neg_develop) and beta (b_neg_develop).
          ):

    ## check the given input is consistent.
    if not isinstance(_data_set_, pd.DataFrame):
      raise TypeError('Use pandas dataframe for the input data')
    if len(self.t_rounds) != 2:
      raise TypeError("len(t_rounds) has to be 2")
    if len(self.t_rounds)*Replicates_ != _data_set_.shape[1]:
      raise TypeError("t_rounds are not compatible with data_set")


    self.data_set = _data_set_.copy()
    self.data_set_all = _data_set_.copy()
    self.Replicates_ = Replicates_

    ### Don't change this even if Initial_library_same is not None
    self.Initial_library_same = None
    ### Don't change this even if Initial_library_same is not None
    
    ### Set the initial condition for alpha and beta inference.
    self.a_neg_develop = np.repeat(ini_abneg[0],len(self.t_rounds))
    self.b_neg_develop = np.repeat(ini_abneg[1],len(self.t_rounds))
    ##
    self.input_name = self.data_set.columns[np.arange(0,self.data_set.shape[1],len(self.t_rounds))]
    self.output_name = self.data_set.columns[np.arange(1,self.data_set.shape[1],len(self.t_rounds))]

    ### when the daata size is too big, set "each_time" with Untilwhichvalue_ = np.mean(self.data_set).mean()*5
    if len(self.data_set) > 1e4:
      self.where_to_threshold = "each_time"
      self.Untilwhichvalue_ = np.mean(self.data_set).mean()*5

    ## Trimming the data or not ################
    if self.where_to_threshold == 'non': # not using trimming
      self.nn_total = self.data_set.sum(axis=0).values
    elif self.where_to_threshold == 'each_time': ## it uses a scaling for theta to reduce the computational cost. (Please refer to the method section of the manuscript.)
      ## theta(t) = - log[sum(e(a_i t + b_i)) * Ntot(t) / sum(n_i(t))], where sum is taken over only for thresholded data. (2022/6/28)
      self.nn_total = self.data_set.sum(axis=0).values
      self.data_set = self.thresholding_data(self.data_set)
    else:
      raise TypeError("where_to_threshold should be non or each_time")


    self.firsttime_ = 'yes'
    
    ## The number of the loops for the two-step method.
    self.repetition_num_negbin = negbin_iterate

    ## Compute the lambda (the expected value of the counts using "lambda_when_two_time_dataset".)
    self.lambda_set_all = self.lambda_when_two_time_dataset(_data_set_.copy())
    ## Consider only those not including 0 counts.    
    self.lambda_set_all_nonzero = self.lambda_set_all[(self.lambda_set_all == 0).sum(axis=1) == 0]

    ## Compute the lambda (the expected value of the counts using "lambda_when_two_time_dataset") from the trimmed data. 
    self.lambda_set = self.lambda_when_two_time_dataset(self.data_set.copy())
    ## Consider only those not including 0 counts.    
    self.lambda_set_nonzero = self.lambda_set[(self.lambda_set == 0).sum(axis=1) == 0]
    self.data_set_nonzero = self.data_set[(self.lambda_set == 0).sum(axis=1) == 0]

    if self.Inference_type == 'Pois':
      print("For the data with two-time points with replicates, Poisson inference is somewhat trivial. We do not include it here. You may use 'fit' instead of 'fit_for_two_time_dataset'.")
    if self.Inference_type == 'Negbin':

      How_many_where_to_take = Replicates_ ## The number of replicates.

      ### FIRST LOOP #########
      ### FIRST LOOP #########
      
      ## Estimate r(lambda) for each replicate and for the time point t=1. 
      for self.where_to_take_lambda in range(How_many_where_to_take): # a loop for specify which replicate we consider
        self.R_inference_remove = 0 # Specify that we will remove the time point t=0, so that we focus on r(lambda) for t=1.
        # Estimate r(lmabda): lambda=self.x_dum_0__, r=self.biased_r_obtained_0__
        self.x_dum_0__, self.biased_r_obtained_0__, _, _, _ = self.r_fitting_from_lambda_and_counts(
                                                                                        self.lambda_set_nonzero.values.copy(),
                                                                                        self.data_set_nonzero.values.copy(),
                                                                                        self.a_neg_develop.copy(), ## This is used only for the initial condition of inference for r
                                                                                        self.b_neg_develop.copy())  ## This is used only for the initial condition of inference for r
        if self.where_to_take_lambda == 0:
          self.x_dum_0 = self.x_dum_0__
          self.biased_r_obtained_0 = self.biased_r_obtained_0__
        else:
          self.x_dum_0 =np.concatenate((self.x_dum_0,self.x_dum_0__)) ## concatenate the estimation for all replicates
          self.biased_r_obtained_0 = np.concatenate((self.biased_r_obtained_0,self.biased_r_obtained_0__)) ## concatenate the estimation for all replicates
        
      _dummy_ = pd.concat((pd.Series(self.x_dum_0), pd.Series(self.biased_r_obtained_0)),axis=1).sort_values(by=[0]) ## change the order in ascending order of lambda
      self.x_dum_0 = _dummy_.values[:,0]
      self.biased_r_obtained_0 = _dummy_.values[:,1]
      y = np.log(self.biased_r_obtained_0)
      X = np.log(self.x_dum_0).reshape(1, -1).transpose()
      fit_func = ConstrainedLinearRegression().fit(X, y, max_coef=[1], min_coef=[0]) ## Fit a linear function to the estimated r(lambda), where all replicate data are combined.
      Fit_func_0 = np.poly1d([fit_func.coef_[0],fit_func.intercept_])
      

      ## The same as above, but this time, focus on the time point t=0.
      for self.where_to_take_lambda in range(How_many_where_to_take): # a loop for specify which replicate we consider
        self.R_inference_remove = 1 # Specify that we will remove the time point t=1, so that we focus on r(lambda) for t=0.
        # Estimate r(lmabda): lambda=self.x_dum_0__, r=self.biased_r_obtained_0__
        self.x_dum_1__, self.biased_r_obtained_1__, _, _, _ = self.r_fitting_from_lambda_and_counts(
                                                                                        self.lambda_set_nonzero.values.copy(),
                                                                                        self.data_set_nonzero.values.copy(),
                                                                                        self.a_neg_develop.copy(), ## This is used only for the initial condition of inference for r
                                                                                        self.b_neg_develop.copy()) ## This is used only for the initial condition of inference for r
        if self.where_to_take_lambda == 0:
          self.x_dum_1 = self.x_dum_1__
          self.biased_r_obtained_1 = self.biased_r_obtained_1__
        else:
          self.x_dum_1 =np.concatenate((self.x_dum_1,self.x_dum_1__)) ## concatenate the estimation for all replicates
          self.biased_r_obtained_1 = np.concatenate((self.biased_r_obtained_1,self.biased_r_obtained_1__)) ## concatenate the estimation for all replicates


      _dummy_ = pd.concat((pd.Series(self.x_dum_1), pd.Series(self.biased_r_obtained_1)),axis=1).sort_values(by=[0]) ## arange the order in ascending order of lambda
      self.x_dum_1 = _dummy_.values[:,0]
      self.biased_r_obtained_1 = _dummy_.values[:,1]
      y = np.log(self.biased_r_obtained_1)
      X = np.log(self.x_dum_1).reshape(1, -1).transpose()
      fit_func = ConstrainedLinearRegression().fit(X, y, max_coef=[1], min_coef=[0]) ## Fit a linear function to the estimated r(lambda), where all replicate data are combined.
      Fit_func_1 = np.poly1d([fit_func.coef_[0],fit_func.intercept_])  ## make a fitting function in the form of poly1d

      rr_min_0 = min(self.x_dum_0) ## This was only for development. Please ignore.
      rr_max_0 = max(self.x_dum_0) ## This was only for development. Please ignore.
      rr_min_1 = min(self.x_dum_1) ## This was only for development. Please ignore.
      rr_max_1 = max(self.x_dum_1) ## This was only for development. Please ignore.
      
      Unbiased_r_func_0 = list(range(self.repetition_num_negbin+1)) ## Make a list to store the fitting function for t=1 
      Unbiased_r_func_1 = list(range(self.repetition_num_negbin+1)) ## Make a list to store the fitting function for t=0 
      Unbiased_r_func_0[0] = Fit_func_0 ## Store the first fitting function
      Unbiased_r_func_1[0] = Fit_func_1 ## Store the first fitting function
      Synthetic_r_func_0 = list(range(self.repetition_num_negbin+1)) ## Make a list to store the fitting function for the synthetic data with t=1 
      Synthetic_r_func_1 = list(range(self.repetition_num_negbin+1)) ## Make a list to store the fitting function for the synthetic data with t=0
      
      ### SECOND LOOP AND LATER #########
      ### SECOND LOOP AND LATER #########
      rep_NB_i=-1
      for rep_NB_i in range(self.repetition_num_negbin-1):

        ##########################################################
        ### Create synthetic data to perform unbias procedure. 
        ##########################################################
        #### first focus on the t=0
        ## obtain lambda value
        _lambda_dummy_0 = self.lambda_set_nonzero.loc[:,self.input_name].copy().values 
        ## compute the corresopnding r value
        rr_synthesis_0 = self.rr_removeoutside(_lambda_dummy_0.flatten(),rr_min_0,rr_max_0,Unbiased_r_func_0[rep_NB_i]).reshape((_lambda_dummy_0.shape[0],_lambda_dummy_0.shape[1])) 
        _dummy_ = np.arange(len(_lambda_dummy_0.flatten()))
        for _lam_, _r_, sample_i in zip(_lambda_dummy_0.flatten(),rr_synthesis_0.flatten(), np.arange(len(_lambda_dummy_0.flatten()))):
          # sampling count data based on negative binomial distribution.
          _dummy_[sample_i] = self.rng.negative_binomial(n=_r_,p=_r_/(_r_+_lam_)) 
          # it continues until it's not 0.
          while _dummy_[sample_i] == 0: 
            _dummy_[sample_i] = self.rng.negative_binomial(n=_r_,p=_r_/(_r_+_lam_))
        # this is the obtained synthetic counts.
        data_set_synthesis_0 = pd.DataFrame(_dummy_.reshape((_lambda_dummy_0.shape[0],_lambda_dummy_0.shape[1])),index=self.lambda_set_nonzero.index,columns=self.input_name) 
        # if the initial library is shared, this condition is imposed to the syntehtic dataset as well. 
        if Initial_library_same == "same": 
          for ii_rep in range(Replicates_):
            if ii_rep != 0:
              data_set_synthesis_0.iloc[:,ii_rep] = data_set_synthesis_0.iloc[:,0]
              
        #### then focus on the t=1
        ## obtain lambda value
        _lambda_dummy_1 = self.lambda_set_nonzero.loc[:,self.output_name].copy().values ## obtain lambda value
        ## compute the corresopnding r value
        rr_synthesis_1 = self.rr_removeoutside(_lambda_dummy_1.flatten(),rr_min_1,rr_max_1,Unbiased_r_func_1[rep_NB_i]).reshape((_lambda_dummy_1.shape[0],_lambda_dummy_1.shape[1]))
        _dummy_ = np.arange(len(_lambda_dummy_1.flatten()))
        for _lam_, _r_, sample_i in zip(_lambda_dummy_1.flatten(),rr_synthesis_1.flatten(), np.arange(len(_lambda_dummy_1.flatten()))):
          # sampling count data based on negative binomial distribution.
          _dummy_[sample_i] = self.rng.negative_binomial(n=_r_,p=_r_/(_r_+_lam_))
          # it continues until it's not 0.
          while _dummy_[sample_i] == 0:
            _dummy_[sample_i] = self.rng.negative_binomial(n=_r_,p=_r_/(_r_+_lam_))
        # this is the obtained synthetic counts.
        data_set_synthesis_1 = pd.DataFrame(_dummy_.reshape((_lambda_dummy_1.shape[0],_lambda_dummy_1.shape[1])),index=self.lambda_set_nonzero.index,columns=self.output_name)

        # this is the obtained synthetic counts.
        self.data_set_synthesis = pd.concat((data_set_synthesis_0,data_set_synthesis_1),axis=1)
        # this is just to change the order in the columns. 
        self.data_set_synthesis = self.data_set_synthesis.loc[:,self.lambda_set_nonzero.columns]


        ### if trimming condition is "each_time", we also perform trimming here for the synthetic data using thresholding_data.
        if self.where_to_threshold == 'each_time':
          self.data_set_synthesis = self.thresholding_data(self.data_set_synthesis.copy())

        ### estimate lambda (the expected value of counts) based on the synthetic dataset using lambda_when_two_time_dataset.
        self.lambda_set_synthesis = self.lambda_when_two_time_dataset(self.data_set_synthesis.copy())
        ### Remove the lines that have a 0 count for lambda. 
        self.lambda_set_synthesis_nonzero = self.lambda_set_synthesis[(self.lambda_set_synthesis == 0).sum(axis=1) == 0]
        ### Remove the lines that have a 0 count for lambda in synthetic dataset. 
        self.data_set_synthesis_nonzero = self.data_set_synthesis[(self.lambda_set_synthesis == 0).sum(axis=1) == 0]

        ######## for t=1
        ######## for t=1
        ## Estimate r(lambda) for each replicate and for the time point t=1 for this synthetic data. 
        for self.where_to_take_lambda in range(How_many_where_to_take): # a loop for specify which replicate we consider
          # This specifies that we will remove the time point t=0, so that we focus on r(lambda) for t=1.
          self.R_inference_remove = 0  
          # estimate r(lambda) for this replicate.
          self.x_dum_synthesis_0__,self.r_obtained_synthesis_0__, _, _, _ = self.r_fitting_from_lambda_and_counts(
                                                                                self.lambda_set_synthesis_nonzero.values.copy(),
                                                                                self.data_set_synthesis_nonzero.values.copy(),
                                                                                self.a_neg_develop.copy(),
                                                                                self.b_neg_develop.copy())    
          if self.where_to_take_lambda == 0:
            self.x_dum_synthesis_0 = self.x_dum_synthesis_0__
            self.r_obtained_synthesis_0 = self.r_obtained_synthesis_0__
          else:
            ## concatenate the estimation for all replicates
            self.x_dum_synthesis_0 =np.concatenate((self.x_dum_synthesis_0,self.x_dum_synthesis_0__))
            ## concatenate the estimation for all replicates
            self.r_obtained_synthesis_0 = np.concatenate((self.r_obtained_synthesis_0,self.r_obtained_synthesis_0__))
        ## combine the results for all the replicates, and rearrange it in ascending order of lambda.
        _dummy_ = pd.concat((pd.Series(self.x_dum_synthesis_0), pd.Series(self.r_obtained_synthesis_0)),axis=1).sort_values(by=[0])
        self.x_dum_synthesis_0 = _dummy_.values[:,0]
        self.r_obtained_synthesis_0 = _dummy_.values[:,1]

        ## Fit a linear function to the obtained function to estimate alpha and beta, based on r = beta * lambda**alpha                                                                              
        Synthetic_r_func_0[rep_NB_i] = np.poly1d(np.polyfit(np.log(self.x_dum_synthesis_0),np.log(self.r_obtained_synthesis_0),6))                                                                                          

        ## Here unbias the estimate of r of the actual dataset, based on the estimation of r for synthetic dataset.
        Unbiased_r_0_estimtate = np.exp( np.log(self.biased_r_obtained_0) + 0.1*( Unbiased_r_func_0[rep_NB_i] - Synthetic_r_func_0[rep_NB_i])(np.log(self.x_dum_0)))

        y = np.log(Unbiased_r_0_estimtate)
        X = np.log(self.x_dum_0).reshape(1, -1).transpose()
        fit_func = ConstrainedLinearRegression().fit(X, y, max_coef=[1], min_coef=[0])
        ## Store the unbiased estimate of alpha and beta.
        Unbiased_r_func_0[rep_NB_i+1] = np.poly1d([fit_func.coef_[0],fit_func.intercept_]) 
        
        ######## for t=0
        ######## for t=0
        ## Estimate r(lambda) for each replicate and for the time point t=0 for this synthetic data. 
        for self.where_to_take_lambda in range(How_many_where_to_take): # a loop for specify which replicate we consider
          # This specifies that we will remove the time point t=0, so that we focus on r(lambda) for t=1.
          self.R_inference_remove = 1
          # estimate r(lambda) for this replicate.
          self.x_dum_synthesis_1__,self.r_obtained_synthesis_1__, _, _, _ = self.r_fitting_from_lambda_and_counts(
                                                                                self.lambda_set_synthesis_nonzero.values.copy(),
                                                                                self.data_set_synthesis_nonzero.values.copy(),
                                                                                self.a_neg_develop.copy(),
                                                                                self.b_neg_develop.copy())   
          if self.where_to_take_lambda == 0:
            self.x_dum_synthesis_1 = self.x_dum_synthesis_1__
            self.r_obtained_synthesis_1 = self.r_obtained_synthesis_1__
          else:
            ## concatenate the estimation for all replicates
            self.x_dum_synthesis_1 =np.concatenate((self.x_dum_synthesis_1,self.x_dum_synthesis_1__))
            ## concatenate the estimation for all replicates
            self.r_obtained_synthesis_1 = np.concatenate((self.r_obtained_synthesis_1,self.r_obtained_synthesis_1__))
        ## combine the results for all the replicates, and rearrange it in ascending order of lambda.
        _dummy_ = pd.concat((pd.Series(self.x_dum_synthesis_1), pd.Series(self.r_obtained_synthesis_1)),axis=1).sort_values(by=[0])
        self.x_dum_synthesis_1 = _dummy_.values[:,0]
        self.r_obtained_synthesis_1 = _dummy_.values[:,1]
                                                                                
        ## Fit a linear function to the obtained function to estimate alpha and beta, based on r = beta * lambda**alpha                                                                              
        Synthetic_r_func_1[rep_NB_i] = np.poly1d(np.polyfit(np.log(self.x_dum_synthesis_1),np.log(self.r_obtained_synthesis_1),6))

        ## Here unbias the estimate of r of the actual dataset, based on the estimation of r for synthetic dataset.
        Unbiased_r_1_estimtate = np.exp( np.log(self.biased_r_obtained_1) + 0.1* (Unbiased_r_func_1[rep_NB_i](np.log(self.x_dum_1)) - Synthetic_r_func_1[rep_NB_i](np.log(self.x_dum_1))))

        y = np.log(Unbiased_r_1_estimtate)
        X = np.log(self.x_dum_1).reshape(1, -1).transpose()
        fit_func = ConstrainedLinearRegression().fit(X, y, max_coef=[1], min_coef=[0])
        ## Store the unbiased estimate of alpha and beta.
        Unbiased_r_func_1[rep_NB_i+1] = np.poly1d([fit_func.coef_[0],fit_func.intercept_])


      ## Finally, average the obtained fitting function Unbiased_r_func_0 and Unbiased_r_func_1 over average_length steps:
      if rep_NB_i > average_length:
        i_count_ = 0
        Unbiased_r_func_0_ave = np.poly1d(Unbiased_r_func_0[rep_NB_i+2-average_length])
        Unbiased_r_func_1_ave = np.poly1d(Unbiased_r_func_1[rep_NB_i+2-average_length])
        i_count_ += 1
        for iiiii in range(rep_NB_i+3-average_length,rep_NB_i+2):
          Unbiased_r_func_0_ave += np.poly1d(Unbiased_r_func_0[iiiii])
          Unbiased_r_func_1_ave += np.poly1d(Unbiased_r_func_1[iiiii])
          i_count_ += 1
        Unbiased_r_func_0_ave = np.poly1d(Unbiased_r_func_0_ave/i_count_)
        Unbiased_r_func_1_ave = np.poly1d(Unbiased_r_func_1_ave/i_count_)
      else:
        Unbiased_r_func_0_ave = 0
        Unbiased_r_func_1_ave = 0
      return Unbiased_r_func_0_ave,Unbiased_r_func_1_ave,self.lambda_set_all_nonzero,[rr_min_0,rr_max_0],[rr_min_1,rr_max_1]
      ########################################################################################
      ### Unbiased_r_func_0_ave = obtained alpha and beta for t=1
      ### Unbiased_r_func_1_ave = obtained alpha and beta for t=0
      ### self.lambda_set_all_nonzero = lambda values (the expected value of counts)
      ### [rr_min_0,rr_max_0],[rr_min_1,rr_max_1] -> Please ignore.
      ########################################################################################



  """
  ######### For two-time data set with replicates. #######################################################
  12_dataset_fitting: 
  This function is to infer alpha, beta, "a" and log(rho0) from count data 
    Input data  : count data
    Output data : obtained alpha and beta for t=1
                  obtained alpha and beta for t=0
                  lambda values (the expected value of counts)
  ########################################################################################################
  """
  def _12_dataset_fitting(self,
          Initial_library_same_dict, ## None if the initial libaray is shared among replicates, "same" otherwise. 
          Num_rep,   ## the number of replicated experiments for each dataset
          Data_name, ## the names of the data text files
          Data_,     ## a list storing count data for each "Data_name"
          Num_core_ = 8 ## number of cores to use.
          ):
            
    try:
      os.mkdir("_12_dataset_fitting")
    except:
      pass

    t_rounds = self.t_rounds
    for i in range(12): # loop for the dataset.
      Initial_library_same = Initial_library_same_dict[i] # if the initial library is the same among replicates.
      print(Data_name[i],Initial_library_same)  
      for i_cross in range(Num_rep[i]): 
        ## loop for the cross validation. "i_cross" will be used for the test, and the others will be used for the training.
        ## prepare a random number generator object 
        seed_num = 123412 
        rng = np.random.RandomState(seed_num)
        
        ## make a folder to save the obtained results.
        save_folder = "_12_dataset_fitting/" + Data_name[i] + '_Repleave'+  str(i_cross)
        try:
          os.mkdir(save_folder)
        except:
          pass
        try:
          os.mkdir(save_folder + "/test")
        except:
          pass
    
        ## Data_column_name stores the names of columns for training, while Data_column_name stores the names of columns for testing
        Data_column_name = []
        Data_column_name_test = []
        for iii in range(Num_rep[i]):
          if iii != i_cross:
            Data_column_name += ["input" + str(iii+1)]
            Data_column_name += ["output" + str(iii+1)]
          else:
            Data_column_name_test  += ["input" + str(iii+1)]
            Data_column_name_test  += ["output" + str(iii+1)]
    
        ## We will use the same variants as those used for the other 5 algorithms for the training and testing.
        _training_data = pd.read_table("./5algorithms_results/" + Data_name[i] + "_test" + str(i_cross+1) + "_training_data.txt",sep =  " ")
        data_set = Data_[i].loc[_training_data[_training_data["input_above_threshold"]].index,Data_column_name]
        data_set_test = Data_[i].loc[ _training_data[_training_data["input_above_threshold"]].index,Data_column_name_test]
    
        ## the number of replicates
        Replicates_ = int(len(Data_column_name)/2)
    
        ## Here, using the trainig dataset, we will estimate alpha and beta for t=0 (Fit_remove1_train) and t=1 (Fit_remove0_train) using fit_for_two_time_dataset from ACIDES. 
        ## We will also estimate the values of lambda (lambda_set_train), which is the exected value of the counts.
        import ACIDES_module
        reload(ACIDES_module)
        from ACIDES_module import ACIDES
        self = ACIDES(Inference_type='Negbin',
                         theta_term_yes='yes',
                         t_rounds=t_rounds,
                         folder_name = save_folder,
                         random_num=rng,
                         para_n_jobs = Num_core_,
                         where_to_threshold = 'non')
        Fit_remove0_train, Fit_remove1_train, lambda_set_train, rr_0, rr_1 = self.fit_for_two_time_dataset(_data_set_= data_set, Replicates_= Replicates_,Initial_library_same=Initial_library_same)      
    
        ## Next, using the testing dataset, we will estimate the values of lambda (lambda_set_test), which is the exected value of the counts.
        import ACIDES_module
        reload(ACIDES_module)
        from ACIDES_module import ACIDES
        self = ACIDES(Inference_type='Negbin',
                         theta_term_yes='yes',
                         t_rounds=t_rounds,
                         folder_name = save_folder + "/test",
                         random_num=rng,
                         para_n_jobs = Num_core_,
                         where_to_threshold = 'non')
        _, _, lambda_set_test, _, _ = self.fit_for_two_time_dataset(_data_set_= data_set_test, Replicates_= 1, Initial_library_same=Initial_library_same, negbin_iterate=1)      
        
        ## For the dataset with i==9, labeling of wildetype uses a different convention from the others, so we change it here.
        if i == 9:
          WT_index = Data_[i][Data_[i]["WT"] == True].index[0]
        else:
          WT_index = Data_[i][Data_[i]["WT"]].index[0]
    
        ############################################################
        ## Below, based on ACIDES results, we will estimate the enrichment and its predicted standard deviation for each variant using resampling. 
        ## Please refer to METHOD section of the manuscript for more details.
        ############################################################
        
        percentile_size = 10000 ## The number of the samples sizes for the resampling.
        
        ## We consider the variants that appear both in training and test datasets.
        lambda_set = pd.concat((lambda_set_train,lambda_set_test),axis=1)
        lambda_set = lambda_set[lambda_set.isna().sum(axis=1) == 0]
        
        ## In enrich_test_train, we will store the enrichment mean and standard deviation for both training ("E_tr_mean","E_tr_std") and testing ("E_tes_mean","E_tes_std").
        enrich_test_train = pd.DataFrame(index = lambda_set.index, columns=["E_tr_mean","E_tr_std","E_tes_mean","E_tes_std"])
            
        
        ## A function to compute the value of r from the fitting function (alpha and beta).
        def rr_removeoutside(xx,x_min,x_max,_Fit_func_):
          return_value = np.exp(_Fit_func_(np.log(xx)))
          return(return_value)
        
        _parallel_ = Num_core_ ## The number of cores to be used. 
        if isinstance(_parallel_, int):
          ## this is a function to estimate (for iii-th variant) the enrichment mean and standard deviation for both training and test dataset 
          def forparallel(iii): ## iii is a variant index.
            first_ = 0
            for jjj in range(Num_rep[i]): ## loop for replicates
              ###################
              #### INPUT ########
              ###################        
              ## input lambda for wild type
              _lam_wt_input_original = lambda_set.loc[WT_index].loc["input" + str(jjj+1)]
              ## input lambda for iii-th variant
              _lam_input_original = lambda_set.iloc[iii].loc["input" + str(jjj+1)]
              if Initial_library_same == None:
                ## If the initial libraries are not common for all the replicates, we will perform the resampling, otherwise keep the lambda as it is.
                ## input r value for wildtype
                _r_wt_input = rr_removeoutside(np.array([_lam_wt_input_original]),rr_1[0],rr_1[1],Fit_remove1_train)[0]
                ## input r value for iii-th variant
                _r_input = rr_removeoutside(np.array([_lam_input_original]),rr_1[0],rr_1[1],Fit_remove1_train)[0]
                ## input resampling for iii-th variant
                _lam_input = rng.negative_binomial(n=_r_input,p=_r_input/(_r_input+_lam_input_original),size=percentile_size)
                ## remove the 0 counts
                _lam_input = _lam_input[_lam_input > 0]
                ## Continue resampling for iii-th variant until all of the samples have counts larger than 0.
                while len(_lam_input) < percentile_size:
                  _lam_input_dummy = rng.negative_binomial(n=_r_input,p=_r_input/(_r_input+_lam_input_original),size=percentile_size)
                  _lam_input_dummy = _lam_input_dummy[_lam_input_dummy>0]
                  _lam_input = np.concatenate((_lam_input,_lam_input_dummy))
                _lam_input = _lam_input[:percentile_size]
                ## input resampling for wild type
                _lam_wt_input = rng.negative_binomial(n=_r_wt_input,p=_r_wt_input/(_r_wt_input+_lam_wt_input_original),size=percentile_size)
              else:
                ## If the initial libraries are common for all the replicates, we keep the lambda as it is.            
                _lam_input = _lam_input_original
                _lam_wt_input = _lam_wt_input_original
    
              ###################
              #### OUTPUT #######
              ###################        
              ## output lambda for wild type
              _lam_wt_output_original = lambda_set.loc[WT_index].loc["output" + str(jjj+1)]
              ## output lambda for iii-th variant
              _lam_output_original = lambda_set.iloc[iii].loc["output" + str(jjj+1)]
              ## output r value for wildtype
              _r_wt = rr_removeoutside(np.array([_lam_wt_output_original]),rr_0[0],rr_0[1],Fit_remove0_train)[0] 
              ## output r value for iii-th variant
              _r_ = rr_removeoutside(np.array([_lam_output_original]),rr_0[0],rr_0[1],Fit_remove0_train)[0]
              ## output resampling for iii-th variant
              _lam_output = rng.negative_binomial(n=_r_,p=_r_/(_r_+_lam_output_original),size=percentile_size)
              ## remove the 0 counts
              _lam_output = _lam_output[_lam_output > 0]
              ## Continue resampling for iii-th variant until all of the samples have counts larger than 0.
              while len(_lam_output) < percentile_size:
                _lam_output_dummy = rng.negative_binomial(n=_r_,p=_r_/(_r_+_lam_output_original),size=percentile_size)
                _lam_output_dummy = _lam_output_dummy[_lam_output_dummy>0]
                _lam_output = np.concatenate((_lam_output,_lam_output_dummy))
              _lam_output = _lam_output[:percentile_size]
              ## output resampling for wild type
              _lam_wt_output = rng.negative_binomial(n=_r_wt,p=_r_wt/(_r_wt+_lam_wt_output_original),size=percentile_size)
                        
              ## enrichment computation for all the (resampled) samples
              _enri_ = np.log((_lam_output/_lam_wt_output) / (_lam_input/_lam_wt_input))
              if jjj != i_cross: 
                ## This is for training dataset, where we take an average of _enri_            
                if first_ == 0:
                  _enri_total =  _enri_
                else:
                  _enri_total +=  _enri_
                first_ += 1
              else:
                 ## This is for test dataset.        
                _enri_test = _enri_
            return [(_enri_total/first_).mean(), (_enri_total/first_).std(), _enri_test.mean(), _enri_test.std()]
            ###############
            #### Return 
            #### (i) the enrichment mean for training : (_enri_total/first_).mean()
            #### (ii) the enrichment standard deviation for training : (_enri_total/first_).std()
            #### (iii) the enrichment mean for testing : _enri_test.mean()
            #### (iv) the enrichment standard deviation for testing : _enri_test.std()
            ###############
    
          ### perform parallel computations for the function "forparallel" defined above over all variant (iii).  
          _dummy_ = Parallel(n_jobs=_parallel_)(delayed(forparallel)(iii) for iii in range(len(lambda_set)))
          
          enrich_test_train.loc[:,"E_tr_mean"] = np.array(_dummy_)[:,0] ## the enrichment mean for training
          enrich_test_train.loc[:,"E_tr_std"] = np.array(_dummy_)[:,1]  ## the enrichment standard deviation for training
          enrich_test_train.loc[:,"E_tes_mean"] = np.array(_dummy_)[:,2]## the enrichment mean for testing
          enrich_test_train.loc[:,"E_tes_std"] = np.array(_dummy_)[:,3] ## the enrichment standard deviation for testing
        else:
          raise TypeError("has to be an integer.")
    
        enrich_test_train = enrich_test_train.astype("float")
        ## Compute the z score:
        enrich_test_train["ACIDES2_z_score"] =   (enrich_test_train["E_tr_mean"] - enrich_test_train["E_tes_mean"])/np.sqrt(enrich_test_train.loc[:,"E_tr_std"]**2 + enrich_test_train.loc[:,"E_tes_std"]**2)
        
        Score_all_dummy = pd.DataFrame(index = range(len(_training_data)),columns=["ACIDES_z"])
        Score_all_dummy.loc[enrich_test_train.index,"ACIDES_z"] = enrich_test_train["ACIDES2_z_score"]
        if i_cross == 0:
          Score_all_0 = Score_all_dummy.copy()
        else:
          ## concatenate all the z-scores over all the replicates.
          Score_all_0 = pd.concat((Score_all_0,Score_all_dummy.copy()),axis=0) 
        
        ## This is to make the correspondence between the indices between the ones obtained from ACIDES and those obtained from the other 5 algorithms.
        if i_cross == 0:
          _training_data_all = _training_data.copy()
        else:
          _training_data_all = pd.concat((_training_data_all,_training_data.copy()),axis=0)
        
      ## The z-score results for the other 5 algorithms to compare. 
      Directly_download_q_score = pd.read_table("./5algorithms_results/" + Data_name[i] + "_leaveoneout_zscore.txt",sep =  " ")
    
      ## reindex Score_all_0 and _training_data_all
      Score_all_0.index = range(len(Score_all_0))
      _training_data_all.index = range(len(_training_data_all))
      
      ## combine Score_all_0 and _training_data_all
      Score_all = pd.concat((Score_all_0,_training_data_all),axis=1)
      ## pick only "test_rep_ok" for ACIDES_z. In Directly_download_q_score, this is already done.
      Directly_download_q_score["ACIDES_z"] = Score_all[Score_all["test_rep_ok"]]["ACIDES_z"].astype("float").values
        
      ## Check if there are no NaN. 
      _Dummy_ = Directly_download_q_score[ ~((Directly_download_q_score["ACIDES_z"].isna()) | (Directly_download_q_score["MioA"].isna())) ]
      
      np.savetxt(save_folder + "/ACIDES_Quant_mean_std.txt",[_Dummy_["ACIDES_z"].std()]) ## ACIDES z-score std
      np.savetxt(save_folder + "/DIM_Quant_mean_std.txt",[_Dummy_["MioA"].std()])        ## DimSum z-score std
      np.savetxt(save_folder + "/naive_Quant_mean_std.txt",[_Dummy_["naive"].std()])     ## S.d. based model z-score std
      np.savetxt(save_folder + "/br_Quant_mean_std.txt",[_Dummy_["br"].std()])           ## Bayes-reg model z-score std
      np.savetxt(save_folder + "/cbe_Quant_mean_std.txt",[_Dummy_["cbe"].std()])         ## Count-based model z-score std
      np.savetxt(save_folder + "/ire_Quant_mean_std.txt",[_Dummy_["ire"].std()])         ## Enrich2 z-score std
    
      Theory_quant = norm.ppf(np.linspace(0.001,0.999,100),loc=0,scale=1)        ## theoreticala quantile
      Quant_ = np.quantile(_Dummy_["ACIDES_z"],np.linspace(0.001,0.999,100))     ## ACIDES quantile
      Quant_dim = np.quantile(_Dummy_["MioA"],np.linspace(0.001,0.999,100))      ## DimSum quantile
      Quant_naive = np.quantile(_Dummy_["naive"],np.linspace(0.001,0.999,100))   ## S. d. based model quantile
      Quant_br = np.quantile(_Dummy_["br"],np.linspace(0.001,0.999,100))         ## Bayes-reg model quantile
      Quant_cbe = np.quantile(_Dummy_["cbe"],np.linspace(0.001,0.999,100))       ## Count_based model quantile
      Quant_ire = np.quantile(_Dummy_["ire"],np.linspace(0.001,0.999,100))       ## Enrich2 quantile
    
      np.savetxt(save_folder + "/ACIDES_R2.txt",[r2_score(Theory_quant,Quant_)]) ## R2 score for ACIDES quantile
      np.savetxt(save_folder + "/DIM_R2.txt",[r2_score(Theory_quant,Quant_dim)]) ## R2 score for DimSum quantile
      np.savetxt(save_folder + "/naive_R2.txt",[r2_score(Theory_quant,Quant_naive)]) ## R2 score for S. d. based model quantile
      np.savetxt(save_folder + "/br_R2.txt",[r2_score(Theory_quant,Quant_br)]) ## R2 score for Bayes-reg model quantile
      np.savetxt(save_folder + "/cbe_R2.txt",[r2_score(Theory_quant,Quant_cbe)]) ## R2 score for Count_based model quantile
      np.savetxt(save_folder + "/ire_R2.txt",[r2_score(Theory_quant,Quant_ire)]) ## R2 score for Enrich2 quantile
      
  """
  ######### For two-time data set with replicates. #######################################################
  Figure_12datasets_6method_comparison: 
  This function is to plot a figure based on 12 dataset analysis
    Output data : Figure to plot (i) inverse standard deviation of z-score and (ii) R2 score calculated in q-q plot.
  ########################################################################################################
  """
  def Figure_12datasets_6method_comparison(self, 
                                           Num_rep,   ## the number of replicated experiments for each dataset
                                           Data_name  ## the names of the data text files
                                           ):
    
    ## store R2 values for each method
    R2_set = pd.DataFrame(index=range(12),columns=["ACIDES", "DIM", "ire" ,  "cbe", "br", "naive"])
  
    ## store std values for each method
    Std_set = pd.DataFrame(index=range(12),columns=["ACIDES", "DIM", "ire" , "cbe", "br", "naive"])
    
    ## the loop for the dataset.
    for i in range(12):
      i_cross = Num_rep[i]-1 # the last replicate, where the data is saved.
      save_folder = "_12_dataset_fitting/" + Data_name[i] + '_Repleave'+  str(i_cross) 
      
      ### R2 score
      R2_set.loc[i,"ACIDES"] = np.loadtxt(save_folder + "/ACIDES_R2.txt") 
      R2_set.loc[i,"DIM"] = np.loadtxt(save_folder + "/DIM_R2.txt")
      R2_set.loc[i,"naive"] = np.loadtxt(save_folder + "/naive_R2.txt")
      R2_set.loc[i,"br"] = np.loadtxt(save_folder + "/br_R2.txt")
      R2_set.loc[i,"cbe"] = np.loadtxt(save_folder + "/cbe_R2.txt")
      R2_set.loc[i,"ire"] = np.loadtxt(save_folder + "/ire_R2.txt")
    
      ### std score
      Std_set.loc[i,"ACIDES"] = np.loadtxt(save_folder + "/ACIDES_Quant_mean_std.txt")
      Std_set.loc[i,"DIM"] = np.loadtxt(save_folder + "/DIM_Quant_mean_std.txt")
      Std_set.loc[i,"naive"] = np.loadtxt(save_folder + "/naive_Quant_mean_std.txt")
      Std_set.loc[i,"br"] = np.loadtxt(save_folder + "/br_Quant_mean_std.txt")
      Std_set.loc[i,"cbe"] = np.loadtxt(save_folder + "/cbe_Quant_mean_std.txt")
      Std_set.loc[i,"ire"] = np.loadtxt(save_folder + "/ire_Quant_mean_std.txt")
      
    ## Number of data.
    num_data_ = R2_set.shape[0]
  
    ## The number of colors.
    NUM_COLORS = 12
    
    # width for the points between different data.
    width = 0.06
    
    ######################################################    
    ######################################################    
    ## Below is to plot two barplots for R2 and for std. 
    ######################################################    
    ######################################################    

    figure(figsize=(7*0.8, 5*1.1)) 
    rr_comb_1 = np.linspace(0,10+2,6) ## position for the box plot with std.
    rr_comb_2 = np.linspace(1,11+2,6) ## position for the box plot with R2.
    
    ## plot vertical line to separate different algorithms. 
    for iiix in (np.linspace(0,10+2,6)[1:] + np.linspace(1,11+2,6)[:-1])/2:
      plt.axvline(x = iiix, color = 'gray')
  
    ############    
    ## For std 
    ############
    ccc = "red"
    cm = plt.get_cmap('Paired')
    ## box plot
    plt.boxplot(1-abs(1-1/Std_set),positions = rr_comb_1,sym="", widths=0.95,patch_artist=True,
                boxprops=dict(facecolor=ccc, color=ccc),
                capprops=dict(color=ccc),
                whiskerprops=dict(color=ccc),
                flierprops=dict(color=ccc, markeredgecolor=ccc),
                medianprops=dict(color="pink",alpha=1)
                )
    ## For each data point
    for i in range(num_data_):
      plt.plot(rr_comb_1+(i+0.5)*width-num_data_/2*width, 1-abs(1-1/Std_set.iloc[i,:]), "o", color = cm(1.*i/NUM_COLORS), markersize=5)
  
    ############
    ## For R2 
    ############
    ccc = "blue"
    R2_set[R2_set < 0] = 0 ## For those with R2<0, we set it to 0. 
    cm = plt.get_cmap('Paired')
    ## box plot
    plt.boxplot(R2_set,positions = rr_comb_2, sym="", widths=0.95,patch_artist=True,
                boxprops=dict(facecolor=ccc, color=ccc),
                capprops=dict(color=ccc),
                whiskerprops=dict(color=ccc),
                flierprops=dict(color=ccc, markeredgecolor=ccc),
                medianprops=dict(color="lightblue",alpha=1))
    ## For each data point
    for i in range(num_data_):
      plt.plot(rr_comb_2+(i+0.5)*width-num_data_/2*width, R2_set.iloc[i,:], "o", color = cm(1.*i/NUM_COLORS), markersize=5)
    
    plt.xticks(rr_comb_2, ["ACIDES", "DiMSum", "Enrich2", "Count-based", "Bayes-reg", "S.d.-based"], rotation=90)
    plt.tight_layout()
    plt.savefig("R2_1overstd_compare.pdf")
    plt.show()  
  

"""
CLASS FILE
ACIDES_FIGURE_AVERAGE:
This is the class file to compute several quantities after using ACIDES. In __init__, several average quantities will be computed first. 
"""
class ACIDES_FIGURE_AVERAGE():
  def __init__(
          self,
          save_folder, ## main folder to make figure, where the results of ACIDES has to be stored.
          Inference_type, ## "Negbin": inference with negative binomial distribution, "Pois": inference with Poisson distribution
          t_rounds = None, ## the time at which NGS is taken (numpy array)
          n_jobs = 1, ## how many cores to use
          howmany_average = 10, ## Average length: quantities are averaged over the last "howmany_average" iterations of two-step algorithms
          average_end = None ## If an integer is provided, the averaging is stopped at this iteration step. 
      ):

    self.save_folder = save_folder
    self.Inference_type = Inference_type
    self.t_rounds = t_rounds
    self.n_jobs = n_jobs
    self.howmany_average = howmany_average
    
    ### This is to know (i) how many iterations are performed, (ii) how many replicates are used, and (iii) if the initial library is the same in ACIDES.
    with open(self.save_folder + '/save_'+ str(0) +'.pkl', 'rb') as inp:
      dummy_self = pickle.load(inp)
    if average_end == None:
      self.average_end = dummy_self.repetition_num_negbin
    else:
      self.average_end = average_end
    self.Replicates_ = dummy_self.Replicates_
    self.Initial_library_same = dummy_self.Initial_library_same
    dummy_self = None
    
    ### below theta term is averaged to calculate the normalization factor (testBB_1 below). See the caption of Fig.S2 for more detail.
    for rep_NB_i in range(self.average_end-howmany_average,self.average_end):
      with open(save_folder + '/save_'+ str(rep_NB_i) +'.pkl', 'rb') as inp:
        dummy_self = pickle.load(inp)
      if rep_NB_i == self.average_end-howmany_average:
        theta_term = dummy_self.theta_term_saved.copy()
      else:
        theta_term += dummy_self.theta_term_saved.copy()
    theta_term /= float(howmany_average)
    testBB_1 = np.polyfit(dummy_self.t_rounds, theta_term, 1)

    ## Obtain data from the results of fit_after_fixing_parameters
    with open(save_folder + '/save_all.pkl', 'rb') as inp:
      dummy_self_all = pickle.load(inp)
    ## getting the score
    data_all = dummy_self_all.scorea_all.copy()
    ## getting the intercept
    data_all = pd.concat((data_all, dummy_self_all.f0_all),axis=1)
    ## renormalizing the theta term.
    data_all['a_inf'] += testBB_1[0]
    data_all['b_inf'] += testBB_1[1]
    ## making a dictionary.
    dummy_dictionary  = {"data_all" : data_all}
    self.dummy_dictionary = dummy_dictionary  


  """
  Experimental_reliability_:
  A function to compute RR. 
  For the definition of RR, please refer to the Method section of the manuscript. 
  """
  def Experimental_reliability_(self, 
                                all_th=5000, # how many top reliable variants we focus. It's 5000 by default.
                                Numa = 3000, # the number of resampled samples for bootstrapping.
                                seed_num = 123, 
                                Total_num = 50 # how many top performing sequences are considered when computing RR. It's 50 by default.
                                ):

    ### the scores of variants.
    data_ = self.dummy_dictionary['data_all'].copy()
    ### sorting ascending order of the errors predicted by the model, and keep the top 5000 most reliable variants: 
    data_ = data_.sort_values(by=["a_inf_err"]).iloc[:all_th]
    ### sorting descending order of the scores predicted by the model
    data_ = data_.sort_values(by=['a_inf'],ascending=False)
    ### index -> naive rank
    data_.index = np.arange(len(data_.index)) + 1 

    Numb = len(data_)
    
    ### perform resampling from here. 
    rng = np.random.RandomState(seed_num)
    ranking_dummy = np.arange(Numa*Numb).reshape((Numa,Numb))
    for i in range(Numa): ## a loop for each resampled scores. 
      ## resampling
      data_dummy = pd.Series(rng.normal(loc=data_['a_inf'],scale=data_['a_inf_err']/2),index=data_.index)
      try:
        ## compute how the original rank is mapped to this resampled rank. 
        ranking_dummy[i] = data_dummy.sort_values(ascending=False).index.values[:Numb]
      except:
        breakpoint()
    
    ## count the number of overlaps between original naive rank and resampled rank.
    for i in range(1,Total_num+1):
      if i == 1:
        correct_rate = (ranking_dummy[:,0:Total_num] == i).sum(axis=1)
      else:
        correct_rate += (ranking_dummy[:,0:Total_num] == i).sum(axis=1)

    ## print the RR:
    correct_rate_normal = correct_rate / Total_num
    print("RR = ", correct_rate_normal.mean())

    
  
  """
  ranking_probability_bootstrap
  This is a function to estimate the confidence interval of the corrected rank based on 
  a sparse matrix with (columns: mapping from the original naive rank to the resampled naive rank) and (rows: resampled sample's index)
  """
  def ranking_probability_bootstrap(self,
                                    ranking_dummy_sparse, # the sparse matrix
                                    where_to_investigate  # the original naive rank we focus 
                                    ):
    dummy = sparse.find(ranking_dummy_sparse == where_to_investigate)[1] + 1 # focus on the original naive rank "where_to_investigate" and look for its resampled rank 
    if len(dummy) > 0:
      test2 = np.percentile(dummy,[2.5,50,97.5]) ## calculate the 95%-confidence interval.
    else:
      test2 = np.array([np.nan,np.nan,np.nan])
    return test2
    
  """
  Ranking_probability_based_on_ai: 
  This is a function to draw the corrected rank (x-axis is the naive rank, y-axis is the corrected rank.)
  """
  def Ranking_probability_based_on_ai(self, 
                                      all_th,                 # how many top reliable variants we focus. It's 5000 by default.
                                      Numa = 3000,            # the number of resampled samples for bootstrapping.
                                      ranking_where = 100,    # How many (naively estimated) top variants are plotted.
                                      seed_num = 123,         # random seed
                                      use_saved_data = False, # Second time to use this funciton, by setting use_saved_data = True, we can skip some computations and quickly draw the figure.
                                      ):

    thre_name = "_th" + str(all_th) + "_"
    Numb_save = all_th

    ### the scores of variants.
    data_ = self.dummy_dictionary['data_all'].copy()
    ### sorting ascending order of the errors predicted by the model, and keep the top 5000 most reliable variants: 
    data_ = data_.sort_values(by=["a_inf_err"]).iloc[:all_th]
    ### sorting descending order of the scores predicted by the model
    data_ = data_.sort_values(by=['a_inf'],ascending=False)
    ### index -> naive rank
    data_.index = np.arange(len(data_.index)) + 1

    ### Number of sequences we consider.
    Numb = len(data_)
    rng = np.random.RandomState(seed_num)
    if use_saved_data == False: # for the first time, it has to compute the confidence interval of the corrected rank, which takes a little bit of time.
      ranking_dummy = np.arange(Numa*Numb).reshape((Numa,Numb))
      for i in range(Numa): # a loop for each resampled scores.
        ## resampling
        data_dummy = pd.Series(rng.normal(loc=data_['a_inf'],scale=data_['a_inf_err']/2),index=data_.index)
        try:
          ## compute how the original rank is mapped to this resampled rank.
          ranking_dummy[i] = data_dummy.sort_values(ascending=False).index.values[:Numb]
        except:
          breakpoint()
      # change it to sparse matrix.
      ranking_dummy_sparse = sparse.csr_matrix(ranking_dummy) 
      # compute the percentile based on the map between the original rank and the resampled rank (ranking_dummy).       
      percentile_set = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self.ranking_probability_bootstrap)(ranking_dummy_sparse,_) for _ in range(1,1+len(data_))))
      # save the obtained percentile set so that it can be used later without computing it again. (set use_saved_data to True to do so.) 
      np.savetxt(self.save_folder + '/percentile_set_Numb' + str(Numb_save)  + thre_name +'.txt',percentile_set)
    elif use_saved_data == True: # after the first time, we can reload the results, so that the figure can be redrawn quickly.
      percentile_set = np.loadtxt(self.save_folder + '/percentile_set_Numb' + str(Numb_save)  + thre_name + '.txt')

    # sort value based on the order of the median rank.
    percentile_set_pd = pd.DataFrame(percentile_set,index=np.arange(1,len(data_)+1)).sort_values(by=[1])
    percentile_set_2 = percentile_set_pd.values
    where_to = np.arange(ranking_where)
    plt.plot(where_to+1, percentile_set_2[where_to,1], 'go',label='estimated ranking')
    plt.errorbar(where_to+1, percentile_set_2[where_to,1], yerr=(percentile_set_2[where_to,1]-percentile_set_2[where_to,0],percentile_set_2[where_to,2]-percentile_set_2[where_to,1]),linestyle='None',alpha=1,color='green')
    plt.xlabel(r'sequence (ascending order of estimated ranking)',fontsize=14)
    plt.ylabel(r'ranking')
    plt.yscale('log')
    plt.tight_layout()
    # save the figure.
    plt.savefig(self.save_folder + '/Ranking2_vs_estimated_ranking_Numb' + str(Numb_save)  + thre_name + '.png') 
    plt.show()

