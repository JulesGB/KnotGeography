import snappy
import sage

def symmetric_alexander_poly(K):
    alex = K.alexander_polynomial(norm=False)
    alex_sym = alex.shift(-min(alex.exponents()) // 2) 
    if (alex_sym(1) == -1):
        alex_sym *= -1
    return alex_sym

# d-invariant
def d_invariant(T_pq):
    alex_sym = symmetric_alexander_poly(T_pq)
    coeff_dict = alex_sym.dict()
    pos_exponents = [k for k in coeff_dict.keys() if k>0]
    
    d_T = 0
    for j in pos_exponents:
        d_T += j * coeff_dict[j] # j * a_j
        
    return 2 * d_T

# Upsilon(1)
def compute_m(idx, alpha_dict):
    if idx == 0:
        return 0

    k = idx // 2
    if idx % 2 == 0:
        return compute_m(2*k-1, alpha_dict) - 1
    else:
        return compute_m(2*k, alpha_dict) - 2 * (alpha_dict[2*k] - alpha_dict[2*k+1]) + 1

# computes Upsilon(1) for positive torus knots T(p,q)
def upsilon(Tpq, t=1):
    alex = symmetric_alexander_poly(Tpq)
    coeff_dict = alex.dict()
    
    alphas = [alpha_k for alpha_k in coeff_dict.keys()]
    n = len(alphas) - 1
    alpha_dict = {n-i : alpha 
                  for i, alpha in zip(range(len(alphas)), alphas)}

    values = [compute_m(2*i, alpha_dict) - t * alpha_dict[2*i] 
              for i in range(0, n//2)]
    return max(values)

# Convert khovanov.Link's to snappy.Link's
def kh_pairings(khL):
    C = khL.crossings
    idx = {X: k for k, X in enumerate(C)}
    pairings = []
    used = set()

    for c, X in enumerate(C):
        for i in range(4):
            Y, j = X.adjacent[i]
            d = idx[Y]
            a = (c, i)
            b = (d, j)
            if a in used or b in used:
                continue
            used.add(a); used.add(b)
            pairings.append((a, b))
    return pairings

def kh_to_snappy(khL):
    Ckh = khL.crossings
    idx = {X: k for k, X in enumerate(Ckh)}

    Csn = [snappy.Crossing(k) for k in range(len(Ckh))]

    seen = set()
    for c, X in enumerate(Ckh):
        for i in range(4):
            (Y, j) = X.adjacent[i]
            d = idx[Y]
            if (d, j, c, i) in seen:
                continue
            Csn[c][i] = Csn[d][j]
            seen.add((c, i, d, j))

    return snappy.Link(Csn)