"""Where the generating words in mkbsgs.py come from: for consecutive subgroups T ⊃ T' of the ladder, the elements of
T outside T' within 7 HTM (the 7-HTM ball, fulltab/ball.py 7), counted by HTM, with the shortest ones written out as
words.  Checks that mkbsgs.WORDS lists exactly the shortest sets it claims: all of H1' \\ H1 within 6 HTM (24 words)
and the shortest element of Y \\ H1'.  Usage: PYTHONPATH=. .venv/bin/python fulltab/census.py  (after mkbsgs.py and
ball 7; about a minute)."""
import os, sys, numpy as np, model as M, coset as CS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ftlib as F
from mkbsgs import WORDS
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ST = np.load(os.path.join(D, "ball7.npy"), mmap_mode="r"); C = np.load(os.path.join(D, "ball7_cost.npy")); W = F.Witness(7, D)
keys = {}
for n in ("Z", "Y2", "Y", "H1p", "H1", "A1", "F16"):
    sp = CS.Spec(canon=n); keys[n] = sp.keys(np.ascontiguousarray(ST)) == sp.keys(CS.pack_states([M.State()]))[0]
keys["solved"] = np.zeros(len(C), bool); keys["solved"][0] = True
state_of = lambda w: M.apply_seq(M.State(), w).key()
found = {}
for a, b in (("Z", "Y2"), ("Y2", "Y"), ("Y", "H1p"), ("H1p", "H1"), ("H1", "A1"), ("A1", "F16"), ("F16", "solved")):
    idx = np.flatnonzero(keys[a] & ~keys[b])
    if len(idx) == 0:
        print(f"{a} \\ {b}: nothing within 7 HTM"); continue
    by = dict(zip(*[x.tolist() for x in np.unique(C[idx], return_counts=True)])); cmin = min(by)
    words = sorted(" ".join(W.seq(int(i))) for i in idx[C[idx] == cmin]); found[(a, b)] = (cmin, words)
    print(f"{a} \\ {b}: by HTM {by}; the {len(words)} shortest ({cmin} HTM): {', '.join(words[:12])}{' ...' if len(words) > 12 else ''}")
# the claims in mkbsgs.WORDS
h1p = found[("H1p", "H1")]; assert h1p[0] == 6 and {state_of(w) for w in h1p[1]} == {state_of(w) for w in WORDS["H1p"][3:]}, "H1' words are not all of H1' \\ H1 within 6 HTM"
y = found[("Y", "H1p")]; assert y[0] == 7 and state_of("R U2 R2 U2 R2 U2 R") in {state_of(w) for w in y[1]}, "Y's extra word is not a shortest element of Y \\ H1'"
print("mkbsgs.WORDS: H1' = H1 plus all 24 elements of H1' \\ H1 within 6 HTM; Y = H1' plus a shortest element of Y \\ H1' (7 HTM): confirmed")
