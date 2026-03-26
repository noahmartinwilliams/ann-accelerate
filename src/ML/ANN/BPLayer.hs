module ML.ANN.BPLayer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix 
import ML.ANN.ActFunc
import ML.ANN.LLayer
import ML.ANN.MkLayer
import ML.ANN.Types
import Prelude as P

bpLayer :: LLayer -> Optim -> AccMat Double Outp One -> (Layer, AccMat Double Outp One)
bpLayer (LLayer { llprevInput = prevInput, llayer = l@(Layer { lnumInputs = ni, lweights = w, lbiases = b, llspec = lspec})}) (SGDOptim lr) bp = do
    let wT = matTransp w
        x = (w `matMul` prevInput ) `matAdd` b
        deriv = matTransp (dactFuncs lspec (matTransp x))
        onesV = AccMat (use (fromList (Z:.ni:.1) (P.repeat 1.0))) Inp One
        bp' = bp `matMul` (matTransp onesV)
        dw = ((x `matZipMul` deriv) `matZipMul` bp) `matMul` (matTransp onesV)
        w' = w `matSub` (lr `matScale` dw)
        db = (deriv `matZipMul` bp)
        b' = b `matSub` (lr `matScale` db)
        (AccMat bp'' Inp One) = wT `matMul` bp
    (l { lweights = w', lbiases = b'}, AccMat bp'' Outp One)
