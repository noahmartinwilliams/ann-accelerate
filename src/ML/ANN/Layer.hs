{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators #-}
module ML.ANN.Layer (Layer(..), LSpec(), calcLayer, lspecGetNumOutputs, layerGetNumInputs, mkSGDLayer) where

import Data.Array.Accelerate as A
import Prelude as P

import ML.ANN.Mat 
import ML.ANN.Vect
import ML.ANN.ActFuncs

type LSpec = [ActFunc]

data Layer = SGDLayer Int (Mat OutputSize InputSize) (Vect OutputSize) LSpec deriving (Show) -- NumInputs weights bias lspec

mkSGDLayer :: [Double] -> LSpec -> Int -> Int -> Layer
mkSGDLayer randoms lspec numInputs numOutputs = do
    let weightsM = use (fromList (Z:.numOutputs:.numInputs) randoms)
        biasesM = use (fromList (Z:.numOutputs:.1) randoms)
    SGDLayer numInputs (MatOI weightsM) (VectO biasesM) lspec

calcLayer :: Layer -> Acc (Matrix Double) -> Acc (Matrix Double)
calcLayer (SGDLayer _ weights bias lspec) x = do
    let x2 = VectI x
        (VectO output) = applyActFuncs lspec ((weights `mmulv` x2 ) `vaddv` bias)
    output

lspecGetNumOutputs :: LSpec -> Int
lspecGetNumOutputs [] = 0
lspecGetNumOutputs ((Sigmoid x) : rest) = x + (lspecGetNumOutputs rest)

layerGetNumInputs :: Layer -> Int
layerGetNumInputs (SGDLayer num _ _ _ ) = num
