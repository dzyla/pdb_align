import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from numba import jit

# Added numba optimizations to slow loops
@jit(nopython=True)
def _window_pairs(D1:np.ndarray, D2:np.ndarray):
    n1,n2=D1.shape[0], D2.shape[0]
    if n1==0 or n2==0: return [], np.array([])
    swapped=False; A,B,aN,bN=D1,D2,n1,n2
    if aN>bN: A,B,aN,bN=D2,D1,n2,n1; swapped=True
    best_score, best_offset = -np.inf, -1
    scores = np.zeros(bN-aN+1)
    for offset in range(bN-aN+1):
        subB=B[offset:offset+aN, offset:offset+aN]
        score=-np.sum(np.abs(A - subB))
        scores[offset] = score
        if score>best_score: best_score, best_offset = score, offset
    
    pairs = []
    if best_offset<0: return pairs, scores
    if swapped: 
        for i in range(aN):
            pairs.append((best_offset+i, i))
    else: 
        for i in range(aN):
            pairs.append((i, best_offset+i))
    return pairs, scores

@jit(nopython=True)
def _banded_dp_maxscore(S:np.ndarray, gap:float, band:int):
    N,M=S.shape
    pairs = []
    if N==0 or M==0: return pairs, 0.0
    neg=-1e18
    dp=np.full((N+1,M+1), neg); bt=np.zeros((N+1,M+1), dtype=np.int8)
    dp[0,0]=0.0
    for i in range(0,N+1):
        jmin=max(0, i-band); jmax=min(M, i+band)
        if i>0:
            j0=max(0, i-band)
            if dp[i-1,j0]>neg:
                dp[i,j0]=max(dp[i,j0], dp[i-1,j0]-gap); bt[i,j0]=2
        for j in range(jmin, jmax+1):
            if i==0 and j==0: continue
            best,move=neg,0
            if i>0 and j>0:
                cand=dp[i-1,j-1]+S[i-1,j-1]
                if cand>best: best,move=cand,1
            if i>0:
                cand=dp[i-1,j]-gap
                if cand>best: best,move=cand,2
            if j>0:
                cand=dp[i,j-1]-gap
                if cand>best: best,move=cand,3
            dp[i,j]=best; bt[i,j]=move
    i,j=N,M
    while i>0 or j>0:
        move=bt[i,j]
        if move==1: pairs.append((i-1,j-1)); i-=1; j-=1
        elif move==2: i-=1
        elif move==3: j-=1
        else: break
    # Numba doesn't natively support pairs.reverse() on list of tuples well
    # Just return reversed list
    pairs_rev = []
    for k in range(len(pairs)-1, -1, -1):
        pairs_rev.append(pairs[k])
    total=float(dp[N,M])
    return pairs_rev, total


def perform_sequence_alignment(seq1:str, seq2:str, gap_open:float, gap_extend:float):
    if not seq1 or not seq2: return None
    try:
        aligner = PairwiseAligner()
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = gap_open
        aligner.extend_gap_score = gap_extend
        aligner.mode = 'global'
        
        # We only need the best alignment
        alignments = aligner.align(seq1, seq2)
        if not alignments: return None
        a = alignments[0]
        
        class Wrap:
            def __init__(self, aln): 
                self.seqA = aln[0]
                self.seqB = aln[1]
                self.score = aln.score
        return Wrap(a)
    except Exception as e:
        return None

