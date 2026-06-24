module ML.ANN.InfLayer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix as M
import ML.ANN.ActFunc
import ML.ANN.MkLayer
import ML.ANN.Types
import Prelude as P


inferLayer :: Layer -> Acc (Matrix Double) -> Acc (Matrix Double)
inferLayer (Layer { lweights = w, lbiases = b, llspec = lspec }) inp = do
    let inp' = AccMat inp Inp One
        m = (w `matMul` inp') `matAdd` b
        (AccMat m' Outp One) = actFuncs lspec m
    m'
inferLayer (InpLayer { vweights = (AccMat w Outp One) , vbiases = (AccMat b Outp One), vlspec = lspec}) inp = do
    let m = A.zipWith (*) inp w
        m' = A.zipWith (+) m b
        m'' = AccMat m' Outp One
        (AccMat m2 Outp One) = actFuncs lspec m''
    m2
