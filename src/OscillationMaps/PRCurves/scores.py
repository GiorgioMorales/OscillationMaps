"""Computation of F_\beta and values of \lambda that split the PRD curve into two even parts """

import numpy as np
from OscillationMaps.PRCurves.utils import interpolate, iou_score


def _prd_to_f_beta(precision, recall, beta=1, epsilon=1e-10):
		#"""from sajjadi"""
	"""Computes F_beta scores for the given precision/recall values.

	The F_beta scores for all precision/recall pairs will be computed and
	returned.

	For precision p and recall r, the F_beta score is defined as:
	F_beta = (1 + beta^2) * (p * r) / ((beta^2 * p) + r)

	Args:
		precision: 1D NumPy array of precision values in [0, 1].
		recall: 1D NumPy array of precision values in [0, 1].
		beta: Beta parameter. Must be positive. The default value is 1.
		epsilon: Small constant to avoid numerical instability caused by division
						 by 0 when precision and recall are close to zero.

	Returns:
		NumPy array of same shape as precision and recall with the F_beta scores for
		each pair of precision/recall.

	Raises:
		ValueError: If any value in precision or recall is outside of [0, 1].
		ValueError: If beta is not positive.
	"""

	if not ((precision >= 0).all() and (precision <= 1).all()):
		raise ValueError('All values in precision must be in [0, 1].')
	if not ((recall >= 0).all() and (recall <= 1).all()):
		raise ValueError('All values in recall must be in [0, 1].')
	if beta <= 0:
		raise ValueError('Given parameter beta %s must be positive.' % str(beta))

	return (1 + beta**2) * (precision * recall) / (
			(beta**2 * precision) + recall + epsilon)

def prd_to_max_f_beta_pair(precision, recall, beta=8):
	"""Computes max. F_beta and max. F_{1/beta} for precision/recall pairs.

	Computes the maximum F_beta and maximum F_{1/beta} score over all pairs of
	precision/recall values. This is useful to compress a PRD plot into a single
	pair of values which correlate with precision and recall.

	For precision p and recall r, the F_beta score is defined as:
	F_beta = (1 + beta^2) * (p * r) / ((beta^2 * p) + r)

	Args:
		precision: 1D NumPy array or list of precision values in [0, 1].
		recall: 1D NumPy array or list of precision values in [0, 1].
		beta: Beta parameter. Must be positive. The default value is 8.

	Returns:
		f_beta: Maximum F_beta score.
		f_beta_inv: Maximum F_{1/beta} score.

	Raises:
		ValueError: If beta is not positive.
	"""

	if not ((precision >= 0).all() and (precision <= 1).all()):
		raise ValueError('All values in precision must be in [0, 1].')
	if not ((recall >= 0).all() and (recall <= 1).all()):
		raise ValueError('All values in recall must be in [0, 1].')
	if beta <= 0:
		raise ValueError('Given parameter beta %s must be positive.' % str(beta))
	
	f_beta = np.max(_prd_to_f_beta(precision, recall, beta))
	f_beta_inv = np.max(_prd_to_f_beta(precision, recall, 1/beta))
	return f_beta, f_beta_inv

def get_extreme_values(precision,recall):
	return max(precision), max(recall)

def prd_to_median_pair(precision, recall):
		# sort the two lists to compute the area
		ind = np.argsort(recall)
		recall = np.squeeze(recall[ind])
		precision = np.squeeze(precision[ind])
		total_area = np.trapz(precision,recall)
		k = 0
		criterion = False
		while (not criterion):
				k+=1
				sub_prec = np.concatenate([np.linspace(0, precision[k],k), precision[k:]])
				sub_recall = np.concatenate([np.linspace(0,recall[k],k), recall[k:]])
				
				sub_area = np.trapz(sub_prec,sub_recall)
				criterion = (sub_area<=(total_area/2))
				
		return (precision[k]+precision[k-1])/2,(recall[k]+recall[k-1])/2
	
def prd_to_auc(precision, recall):
		# sort the two lists to compute the area
		ind = np.argsort(recall)
		recall = np.squeeze(recall[ind])
		precision = np.squeeze(precision[ind])
		total_area = np.trapz(precision,recall)
		return total_area

def get_precision_at_epsilon(precision, recall, epsilon):
	if max(recall)<epsilon:
		return 0
	else :
		k = 0
		while recall[k]<epsilon:
			k+=1
		t = (epsilon-recall[k-1])/(recall[k]-recall[k-1])
		precision_at_epsilon = (1-t)*precision[k-1]+t*precision[k]
		return precision_at_epsilon

def get_recall_at_epsilon(precision, recall, epsilon):
	if max(precision)<epsilon:
		return 0
	else :
		k = 0
		while precision[k]>epsilon:
			k+=1
		t = (epsilon-precision[k])/(precision[k-1]-precision[k])
		recall_at_epsilon = (1-t)*recall[k]+t*recall[k-1]
		return recall_at_epsilon
	
def get_all_scores(alphas_betas, epsilon=0.05, beta=9):
    precision, recall = alphas_betas[:,0],alphas_betas[:,1]
    
    max_precision, max_recall = get_extreme_values(precision=precision, recall=recall)
    median_precision, median_recall = prd_to_median_pair(precision=precision, recall=recall)
    fbeta_recall, fbeta_precision = prd_to_max_f_beta_pair(precision=precision, recall=recall,beta=beta)
    auc = prd_to_auc(precision, recall)
    precision_at_eps = get_precision_at_epsilon(precision=precision, recall=recall, epsilon=epsilon)
    recall_at_eps = get_recall_at_epsilon(precision=precision, recall=recall, epsilon=epsilon)
    
    dict_metrics = {
                    "max_precision":max_precision,
                    "max_recall":max_recall,
                    "median_precision":median_precision,
                    "median_recall":median_recall,
                    "fbeta_precision":fbeta_precision,
                    "fbeta_recall":fbeta_recall,
                    "auc":auc,
                    "precision_at_eps":precision_at_eps,
                    "recall_at_eps":recall_at_eps
                    }
    return dict_metrics

def get_iou_scores_shift_exp(arrs_dict):
	NB_POINTS_INTERP = 111

	# Compute all the interpolations to then compute the IOU score

	methods_interp = {}

	for method in arrs_dict:
			if methods_interp.get(method) is None:
					methods_interp[method] = {}
			for shift in arrs_dict[method]:
					
					list_interp_shifts = []
					for array in arrs_dict[method][shift]:
							alphas_method = array[:,0] 
							betas_method  = array[:,1]
							alphas_method_interp,_ = interpolate(alphas_method,betas_method,NB_POINTS_INTERP)
							list_interp_shifts.append(alphas_method_interp)
					methods_interp[method][shift] = list_interp_shifts                

	# Compute IOU score
	x = np.linspace(0,1,num=NB_POINTS_INTERP)

	dict_scores = {}

	for method in methods_interp:
			if dict_scores.get(method) is None:
					dict_scores[method] = {}
			for shift in methods_interp[method]:
					dict_scores[method][shift]=[iou_score(methods_interp["ground-truth"][shift][0],interp_method,x) for interp_method in methods_interp[method][shift]]
					

	dict_scores_stats = {}
	for method in dict_scores:
			if dict_scores_stats.get(method) is None:
					dict_scores_stats[method] = {}
			for shift in dict_scores[method]:
					dict_scores_stats[method][shift]= {"mean":np.mean(np.stack(dict_scores[method][shift],axis=0)),
																					"std":np.std(np.stack(dict_scores[method][shift],axis=0))}
			
	return dict_scores_stats

def print_score_latex_grid(dict_scores_stats,methods,shifts):
		print("shift value &"," & ".join(methods), r"\\ \hline\hline ")
		for shift in shifts:
				print(shift," & "," & ".join(f'{dict_scores_stats[method][shift]["mean"]:.2f} +- {dict_scores_stats[method][shift]["std"]:.2e}' for method in methods), r"\\")
		print("\n\n")
		print("shift value &"," & ".join(methods), r"\\ \hline\hline ")
		for shift in shifts:
				print(shift," & "," & ".join(f'{dict_scores_stats[method][shift]["mean"]:.2f}' for method in methods), r"\\")