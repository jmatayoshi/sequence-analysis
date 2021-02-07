import numpy as np
import statsmodels.api as sm


def generate_sequence(seq_length, states, base_rates):
    seq = list(np.random.choice(states,
                                size=seq_length,
                                p=base_rates)
    )
    return seq


def compute_cond_probs(seq, states):
    prev_count = {a: 0 for a in states}
    cond_count = {a: {b: 0 for b in states} for a in states}
    for i in np.arange(1, len(seq)):
        for a in states:
            if seq[i - 1] == a:
                prev_count[a] += 1
                for b in states:
                    if seq[i] == b:
                        cond_count[a][b] += 1
                        break
                break   
    res = []
    for a in states:
        for b in states:
            if prev_count[a] > 0:
                res.append(cond_count[a][b] / prev_count[a])
            else:
                res.append(np.nan)
    return np.array(res).reshape((len(states), len(states)))


def compute_L(seq, A, B):
    A_prev = 0
    B_next_A_prev = 0
    B_next = 0

    for i in np.arange(1, len(seq)):
        if seq[i] == B:
            B_next += 1
        if seq[i - 1] == A:
            A_prev += 1
            if seq[i] == B:
                B_next_A_prev += 1

    P_B_next = B_next / (len(seq) - 1)

    if P_B_next == 1:
        return np.nan
    elif A_prev == 0:
        return np.nan
    else:
        P_B_next_A_prev = B_next_A_prev / A_prev
        return (P_B_next_A_prev - P_B_next) / (1 - P_B_next)
   

def fit_marginal_model(y, X, groups, cov=sm.cov_struct.Exchangeable()):
    """ Fit marginal model to sequential data.

    Parameters
    ----------
    y : 1d numpy array
        Array of dependent variables ("endogeneous" variables)    
    X : 2d numpy array
        Array of independent variables ("exogeneous" variables)
    seq_ind : 1d numpy array
        Array containing cluster/group labels for GEE model
    cov : statsmodels class, optional
        One of the following working dependence structures for GEE model: 
            sm.cov.struct.Independence()
            sm.cov.struct.Exchangeable()
            sm.cov.struct.Autoregressive()

    Returns
    -------
    beta_coef : float
         Coefficient from the single independent variable in the GEE model
    p_val : float
         The p-value returned from a two-tailed t-test on beta_coef
    P_A_B : float
         Estimated probability of transitioning to A given that the starting 
         state is B
    P_A_not_B : float
         Estimated probability of transitioning to A given that the starting 
         state is not B
    """
    md = sm.GEE(
        y, X, groups,
        cov_struct=cov,
        family=sm.families.Binomial()
    )    
    fit_res = md.fit()
    beta_coef = fit_res.params[1]
    p_val = fit_res.pvalues[1]  
    P_A_B = md.predict(fit_res.params, exog=np.array([1, 1]))
    P_A_not_B = md.predict(fit_res.params, exog=np.array([1, 0]))

    return beta_coef, p_val, P_A_B, P_A_not_B

    
def df_to_y_X(df, a, b, min_length=0):
    """ Function for turning a DataFrame of sequential data into the appropriate
    format for GEE model

    Parameters
    ----------
    df : DataFrame
        First column contains a student index, second column an affect state
        Rows are grouped based on the student index and ordered sequentially 
        Example:
            1,CON
            1,FLO
            1,FRU
            2,FLO
            2,FLO
            3,BOR
    a : str/float
        Starting state
    b : str/float
        Ending state
    min_length : int, optional
        Sequences less than min_length are excluded

    Returns
    -------
    y : 1d numpy array
        Array of dependent variables ("endogeneous" variables)    
    X : 2d numpy array
        Array of independent variables ("exogeneous" variables)
    seq_ind : 1d numpy array
        Array containing cluster/group labels for GEE model
    """
    y = []
    X = []
    seq_ind = []
    for i in np.unique(df.iloc[:, 0].values):
        pos = np.flatnonzero(df.iloc[:, 0].values == i)
        if len(pos) >= min_length:
            for j in range(len(pos) - 1):
                if df.iloc[pos[j], 0] == df.iloc[pos[j + 1], 0]:
                    seq_ind.append(df.iloc[pos[j], 0])
                    if df.iloc[pos[j], 1] == a:
                        X.append([1, 1])
                    else:
                        X.append([1, 0])            
                    if df.iloc[pos[j + 1], 1] == b:
                        y.append(1)
                    else:
                        y.append(0)                
    return np.array(y), np.array(X), np.array(seq_ind)


def sequences_to_y_X(seq_list, a, b, min_length=0):
    """ Function for turning a list of sequences into the appropriate
    format for GEE model

    Parameters
    ----------
    seq_list : list of lists
        Each entry in the list is a sequence (list) of transition states
        Example:
            [
                ['A', 'C', 'C', 'B', 'C'],
                ['B', 'C', 'A', 'C'],
                ['C', 'C', 'C', 'B', 'B', 'A']
            ]
    a : str/float
        Starting state
    b : str/float
        Ending state
    min_length : int, optional
        Sequences less than min_length are excluded

    Returns
    -------
    y : 1d numpy array
        Array of dependent variables ("endogeneous" variables)    
    X : 2d numpy array
        Array of independent variables ("exogeneous" variables)
    seq_ind : 1d numpy array
        Array containing cluster/group labels for GEE model
    """
    y = []
    X = []
    seq_ind = []
    for i in range(len(seq_list)):
        curr_seq = seq_list[i]
        if len(curr_seq) >= min_length:
            for j in range(len(curr_seq) - 1):
                seq_ind.append(i)                
                if curr_seq[j] == a:
                    X.append([1, 1])
                else:
                    X.append([1, 0])
                if curr_seq[j + 1] == b:
                    y.append(1)
                else:
                    y.append(0)

    return np.array(y), np.array(X), np.array(seq_ind)


def run_simulations(
        num_trials=10000,
        base_rates=np.array([0.5, 0.5]),
        seq_lengths=np.arange(3, 151)):
    """ Run numerical experiments 
    
    Experiment 1 parameters (results shown in Figures 1 and 2): 
        num_trials=10000,
        base_rates=np.array([0.5, 0.5]),
        seq_lengths=np.arange(3, 151)

    Experiment 2 parameters (results shown in Figure 3): 
        num_trials=10000,
        base_rates=np.array([0.6, 0.2, 0.1, 0.1]),
        seq_lengths=np.arange(3, 151)
    
    Returns
    -------
    Average conditional probabilities for A-->A and A-->B
    GEE estimated conditional probabilities for A-->A and A-->B
    L values for A-->A and A-->B
    GEE \beta_1 values for A-->A and A-->B
    
    """    
    L_AA = []
    L_AB = []
    P_AA = []
    P_AB = []
    gee_beta_AA = []
    gee_beta_AB = []
    gee_P_AA = []
    gee_P_AB = []

    states = np.arange(base_rates.shape[0])

    for seq_len in seq_lengths:
        if seq_len % 5 == 0:
            print('Current sequence length = ' + str(seq_len))

        seq_list = []
        for i in range(num_trials):
            seq_list.append(generate_sequence(seq_len, states, base_rates))
        curr_AA = []
        curr_AB = []
        curr_P_AA = []
        curr_P_AB = []        
        for i in range(len(seq_list)):
            curr_seq = seq_list[i]
            curr_AA.append(compute_L(curr_seq, states[0], states[0]))
            curr_AB.append(compute_L(curr_seq, states[0], states[1]))
            res = compute_cond_probs(curr_seq, states)
            curr_P_AA.append(res[0, 0])
            curr_P_AB.append(res[0, 1])            
        L_AA.append(np.nanmean(curr_AA))
        L_AB.append(np.nanmean(curr_AB))
        P_AA.append(np.nanmean(curr_P_AA))
        P_AB.append(np.nanmean(curr_P_AB))            

        y, X, groups = sequences_to_y_X(seq_list, states[0], states[0])                
        res = fit_marginal_model(y, X, groups, cov=sm.cov_struct.Exchangeable())
        gee_beta_AA.append(res[0])
        gee_P_AA.append(res[2])

        y, X, groups = sequences_to_y_X(seq_list, states[0], states[1])
        res = fit_marginal_model(y, X, groups, cov=sm.cov_struct.Exchangeable())
        gee_beta_AB.append(res[0])
        gee_P_AB.append(res[2])

    return [
        # conditional probabilities
        P_AA, P_AB,
        # GEE estimated conditional probabilities
        gee_P_AA, gee_P_AB,
        # L statistic values
        L_AA, L_AB,
        # GEE \beta_1 coefficients
        gee_beta_AA, gee_beta_AB,
    ]
