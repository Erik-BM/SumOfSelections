"""
Sum of Selections. 
Given a list T of positive integers and a positive integer K. Find if any combination of T sums to K, if so find the combination.
"""

import numpy as np
import argparse

def read_input(filename):
    """
    Read data from input file on form:
    eg.
    5 14 1 5 6 3 10
    where first number is N, second is K and rest is data set.
    args:
        filename :: string
            path to input file
    returns:
        out :: list of tuples.
    """
    out = []
    with open(filename, 'r') as f:
        for l in f:
            try:
                d = l.split()
                n = int(d[0])
                K = int(d[1])
                ts = [int(i) for i in d[2:]]
                if len(ts) != n:
                    raise TypeError('N not equal to length of T')
                out.append([n,K,ts])
            except IndexError:
                pass
    return out

def save_output(filename, data, answer, subset):
    """
    Saves answer and subset in desired format.
    eg. 
    INSTANCE 5 14: 1 5 6 3 10
    YES 
    SELECTION 1[1] 3[4] 10[5]                                            
    
    args:
        filename :: str
            file to save to
        data :: tuple
            input data
        answer :: bool
        subset :: list or None
    returns:
        None
    """
    if answer:
        s = "INSTANCE {0} {1}: {2}\nYES".format(str(data[0]), str(data[1]),' '.join([str(i) for i in sorted(data[2])]))
        if subset:
            s += "\nSELECTION "
            for i,t in subset:
                s += "{0}[{1}] ".format(t,i+1)
    else:
        s = "INSTANCE {0} {1}: {2}\nNO".format(str(data[0]), str(data[1]),' '.join([str(i) for i in sorted(data[2])]))
        
    with open(filename, 'a+') as f:
        f.write(s)
        f.write("\n\n")

def bottom_up(N,K,ts):
    """
    Recursive algorithm.
    args:
        N :: int
            length of ts
        K :: int 
        ts :: list of ints
    returns:
        res :: bool 
            True :: if a subset of ts sums to K
            False :: otherwise
        subset :: list of tuples
            index and value in ts of the subset that sums to K.
    """
    U = np.zeros((N+1,K+1), dtype = bool)
    U[:,0] = 1 # Sum to zero always possible
    
    subset = set([])
    for i,t in enumerate(ts, start = 1):
        for j in range(1, K+1):
            if j >= t:
                U[i,j] = U[i-1,j-t] or U[i-1,j]
            else:
                U[i,j] = U[i-1,j]
    
    res = U[N,K]
    # Find subset by traversing array from solution at (N,K) to (1,1).
    subset = []
    k = K
    n = N
    if res:
        while n > 0 and k > 0:
            if not U[n-1, k]:
                subset.append((n-1,ts[n-1]))
                k -= ts[n-1]
            n -= 1
            
    return res, sorted(subset)

def bottom_up_low_space(N,K,ts):
    """
    Recursive algorithm.
    args:
        N :: int
            length of ts
        K :: int 
        ts :: list of ints
    returns:
        res :: bool 
            True :: if a subset of ts sums to K
            False :: otherwise
        subset :: list of tuples
            index and value in ts of the subset that sums to K.
    """
    
    U = np.zeros(K+1, dtype = int)
    U[0] = 1
    for t in ts:
        j = K
        while j >= t:
            if U[j-t] != 0:
                U[j] = t
            j -= 1
    
    res = U[K] != 0
    subset = []
    k = K
    while U[k] != 0 and k > 0:
        t = U[k]
        i = [i for i, x in enumerate(U[k] == ts) if x]
        for l in i:
            if not (l,t) in subset:
                subset.append((l,t))
        k -= U[k]
    
    return res, sorted(subset)

def top_down_function(N,K,ts):
    """
    Recursive algorithm.
    args:
        N :: int
            length of ts
        K :: int 
        ts :: list of ints
    returns:
        True :: if a subset of ts sums to K
        False :: otherwise
    """
    # Sum to zero always possible.
    if K == 0:
        return True
    # If empty set and K not zero.
    if (N == 0 and K != 0):
        return False
    # If previous element is larger than K, i.e new K < 0
    if (ts[N-1] > K):
        return top_down_function(N-1, K, ts)
    
    return top_down_function(N-1, K, ts) or top_down_function(N-1,K-ts[N-1],ts)

def top_down(N,K,ts):
    """
    Call recursive algorithm and returns result and subset.
    args:
        N :: int
            length of ts
        K :: int 
        ts :: list of ints
    returns:
        res :: bool if a subset of ts sums to K
        subset :: set of element
    """
    res = top_down_function(N,K,ts)
    subset = []
    k = K
    n = N
    if res:
        while n > 0 and k > 0:
            if not top_down_function(n-1, k, ts):
                subset.append((n-1,ts[n-1]))
                k -= ts[n-1]
            n -= 1
            
    return res, sorted(subset)
    

def poly_time(N,K,ts):
    """
    Bonus implementation.
    https://en.wikipedia.org/wiki/Subset_sum_problem#Polynomial_time_approximate_algorithm
    
    args:
        N :: int
            length of ts
        K :: int 
        ts :: list of ints
    returns:
        True :: if a subset of ts sums to K
        False :: otherwise
    """
    p = max(ts).bit_length()
    c = 2**-p
    U = set()
    S = set([0])
    for i in range(N):
        T = set([ts[i] + y for y in S])
        U = T.union(S)
        U = sorted(list(U))
        S = set()
        y = min(U)
        S.add(y)
        for z in U:
            if z <= K and y + c*K/N < z:
                y = z 
                S.add(z)
                
    res = False
    for s in S:
        if (1-c)*K <= s <= K:
            res = True
            
    return res

def main(input_filename, output_filename, function):
    """
    Load dataset and call algorithm specified in function. Finally, save output.
    
    args:
        input_filename :: string
            path to .txt file with input.
        output_filename :: string
            path to output file
        function :: function object with three inputs (N,K,ts).
    return
        None
    """
    
    
    for r in read_input(input_filename):
        N,K,ts = r
        ts = sorted(ts)
        res = function(N,K,ts)
        try:
            c,subset = res
        except TypeError:
            c = res
            subset = None
        save_output(output_filename, r, c, subset)
        

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Sums of Selections')
    parser.add_argument('input_filename', metavar='input', type=str, nargs='?',
                    help='input filename')
    parser.add_argument('output_filename', metavar='output', type=str, nargs='?',
                    help='output filename')
    parser.add_argument('--top', action='store_true', help='recursive top-down')
    parser.add_argument('--bottom', action='store_true', help='iterative bottom-down')
    parser.add_argument('--poly', action='store_true', help='approx poly time algorithm')
    parser.add_argument('--bottom_low', action='store_true', help='iterative bottom-down low space')
    
    args = parser.parse_args()
    
    if args.poly:
        main(args.input_filename, args.output_filename, poly_time)
    if args.top:
        main(args.input_filename, args.output_filename, top_down)
    if args.bottom:
        main(args.input_filename, args.output_filename, bottom_up)
    if args.bottom_low:
        main(args.input_filename, args.output_filename, bottom_up_low_space)
    
    
    
    
    
    
    
