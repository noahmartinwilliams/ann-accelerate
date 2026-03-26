module ML.ANN.InfLayer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix as M
import ML.ANN.ActFunc
import ML.ANN.MkLayer
import ML.ANN.Types
import Prelude as P


inferLayer :: Acc (Matrix Double) -> Layer -> Acc (Matrix Double)
inferLayer inp (Layer { lweights = w, lbiases = b, llspec = lspec }) = do
    let inp' = AccMat inp Inp One
        m = (w `matMul` inp') `matAdd` b
        (AccMat m' One Outp) = actFuncs lspec (matTransp m )
    A.transpose m'
inferLayer inp (InpLayer { vweights = (AccMat w Outp One) , vbiases = (AccMat b Outp One), vlspec = lspec}) = do
    let m = A.zipWith (*) inp w
        m' = A.zipWith (+) m b
        m'' = AccMat (A.transpose m') One Outp
        (AccMat m2 _ _) = actFuncs lspec m''
    A.transpose m2
