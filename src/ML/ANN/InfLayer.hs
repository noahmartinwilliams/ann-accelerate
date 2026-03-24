module ML.ANN.InfLayer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix 
import ML.ANN.MkLayer
import ML.ANN.Types
import Prelude as P

inferLayer :: Acc (Matrix Double) -> Layer -> Acc (Matrix Double)
inferLayer inp (Layer { lweights = w, lbiases = b }) = do
    let inp' = AccMat inp Inp One
        (AccMat m _ _ ) = (w `matMul` inp') `matAdd` b
    m 
