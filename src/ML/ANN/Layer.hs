{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators #-}
module ML.ANN.Layer (Layer(..), LSpec(), calcLayer) where

import Data.Array.Accelerate as A
import Prelude as P

import ML.ANN.Mat 
import ML.ANN.Vect
import ML.ANN.ActFuncs

type LSpec = [ActFunc]

data Layer = SGDLayer (Mat OutputSize InputSize) (Vect OutputSize) LSpec

calcLayer :: Layer -> Acc (Matrix Double) -> Acc (Matrix Double)
calcLayer (SGDLayer weights bias lspec) x = do
    let x2 = VectI x
        (VectO output) = applyActFuncs lspec ((weights `mmulv` x2 ) `vaddv` bias)
    output
