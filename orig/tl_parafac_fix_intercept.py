import numpy as np
import tensorly as tl
import orig.tl_src as tl_src
import warnings

def parafac_fix_intercept(
    tensor, 
    rank,
    fixed_modes=None,
    fix_intercept_mode=-1,
    overweight_mode=-1,
    gamma=0,
    normalize_factors=False, 
    n_iter_max=100,
    tol=1e-8,
    cvg_criterion='abs_rec_error',
    init='svd',
    svd='truncated_svd',
    random_state=None,
    verbose=False,
    return_errors=False,
    weights=None,
    factors=None
):
    if overweight_mode == -1:
        gamma = 0
    
    rank = tl_src.validate_cp_rank(tl.shape(tensor), rank=rank)

    ### initialization
    if (weights is None) or (factors is None):
        weights, factors = tl_src.initialize_cp(tensor, rank, init=init, svd=svd,
                                     random_state=random_state,
                                     normalize_factors=normalize_factors)

    ### check mode to be fixed
    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    if fixed_modes is None:
        fixed_modes = []

    if fixed_modes == list(range(tl.ndim(tensor))): # Check If all modes are fixed
        cp_tensor = CPTensor((weights, factors)) # No need to run optimization algorithm, just return the initialization
        return cp_tensor

    if tl.ndim(tensor)-1 in fixed_modes:
        warnings.warn('You asked for fixing the last mode, which is not supported.\n The last mode will not be fixed. Consider using tl.moveaxis()')
        fixed_modes.remove(tl.ndim(tensor)-1)
    modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    if verbose:
        print('modes_list: {}'.format(modes_list))

    tensor_overweight=tensor+gamma*np.mean(tensor,axis=overweight_mode,keepdims=True)

    # Checkpoint: tensors, factors, weights same up to here

    ### alternating regression
    for iteration in range(n_iter_max):

        if verbose > 1:
            print("Starting iteration", iteration + 1)

        ### main step: alternating regression
        for mode in modes_list:
            if verbose > 1:
                print("Mode", mode, "of", tl.ndim(tensor))

            if mode==fix_intercept_mode:
                # fix the first col of factor mode to all one, and solve for other columns
                pseudo_inverse = tl.tensor(np.ones((rank-1, rank-1)), **tl.context(tensor))
                for i, factor in enumerate(factors):
                    if i != mode:
                        pseudo_inverse = pseudo_inverse * tl.dot(tl.transpose(factor[:,1:]), factor[:,1:])

                Z=tl.tenalg.khatri_rao([factors[i] for i in range(len(factors)) if i!=mode]).T

                if mode!=overweight_mode:
                    mttkrp=(tl.unfold(tensor_overweight,mode)-(np.ones((factors[mode].shape[0],1))@Z[0,:][np.newaxis,:]))@Z[1:,:].T
                else:
                    mttkrp=(tl.unfold(tensor,mode)-(np.ones((factors[mode].shape[0],1))@Z[0,:][np.newaxis,:]))@Z[1:,:].T

                factor = tl.transpose(tl.solve(tl.transpose(pseudo_inverse),
                                      tl.transpose(mttkrp)))

                factor=np.concatenate((np.ones((factor.shape[0],1)),factor),axis=1)
            else:

                pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
                for i, factor in enumerate(factors):
                    if i != mode:
                        pseudo_inverse = pseudo_inverse * tl.dot(tl.transpose(factor), factor)

                if mode!=overweight_mode:
                    mttkrp = tl_src.unfolding_dot_khatri_rao(tensor_overweight, (None, factors), mode)
                else: 
                    mttkrp = tl_src.unfolding_dot_khatri_rao(tensor, (None, factors), mode)

                factor = tl.transpose(tl.solve(tl.transpose(pseudo_inverse),
                                      tl.transpose(mttkrp)))


            if normalize_factors:
                scales = tl.norm(factor, 2, axis=0)
                weights = tl.where(scales==0, tl.ones(tl.shape(scales), **tl.context(factor)), scales)
                factor = factor / tl.reshape(weights, (1, -1))

            factors[mode] = factor

        ### Calculate the current unnormalized error if we need it
        if (tol or return_errors):
            unnorml_rec_error, tensor, norm_tensor = tl_src.error_calc(tensor, norm_tensor, weights, factors, None, None, mttkrp)


            rec_error = unnorml_rec_error / norm_tensor
            rec_errors.append(rec_error)

        if tol:

            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print("iteration {}, reconstruction error: {}, decrease = {}, unnormalized = {}".format(iteration, rec_error, rec_error_decrease, unnorml_rec_error))

                if cvg_criterion == 'abs_rec_error':
                    stop_flag = abs(rec_error_decrease) < tol
                elif cvg_criterion == 'rec_error':
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print("PARAFAC converged after {} iterations".format(iteration))
                    break

            else:
                if verbose:
                    print('reconstruction error={}'.format(rec_errors[-1]))

    cp_tensor = tl_src.CPTensor((weights, factors))
    
    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor