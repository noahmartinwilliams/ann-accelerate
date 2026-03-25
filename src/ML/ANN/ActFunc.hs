module ML.ANN.ActFunc where

import Data.Array.Accelerate as A
import Prelude as P

sigmoid :: Exp Double -> Exp Double
sigmoid x = let one = constant 1.0 in one / (one + (exp (-x)))

relu :: Exp Double -> Exp Double
relu x = A.max (constant 0.0) x

softmax :: Acc (Matrix Double) -> Acc (Matrix Double)
softmax x = do
    let e = A.map (\y -> exp y) x
        s = A.sum (A.flatten e)
        s' = A.the s
    A.map (\y -> y / s') e
