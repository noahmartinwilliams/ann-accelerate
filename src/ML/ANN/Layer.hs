{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators #-}
module ML.ANN.Layer (Layer(..), LLayer(..), LSpec(), calcLayer, lspecGetNumOutputs, layerGetNumInputs, mkSGDLayer, learnLayer, backpropLayer) where

import Data.Array.Accelerate as A
import Prelude as P

import ML.ANN.Mat 
import ML.ANN.Vect
import ML.ANN.ActFuncs
import ML.ANN.Optim

type LSpec = [ActFunc]

data Layer = SGDLayer Int (Mat OutputSize InputSize) (Vect OutputSize) LSpec deriving (Show) -- NumInputs weights bias lspec

data LLayer = LSGDLayer Layer (Vect InputSize) deriving(Show)

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

learnLayer :: Layer -> Acc (Matrix Double) -> (LLayer, Acc (Matrix Double))
learnLayer (SGDLayer numInputs weights biases lspec) input = do
    let x = VectI input
        output = applyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
    ((LSGDLayer (SGDLayer numInputs weights biases lspec) x), (extractVect output))

backpropLayer :: LLayer -> Optim -> Acc (Matrix Double) -> (Layer, Acc (Matrix Double))
backpropLayer (LSGDLayer layer x) (SGD lr) bp = do
    let (SGDLayer numInputs weights biases lspec) = layer
        bp2 = VectO bp
        deriv = dapplyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
        weights2 = weights `msubm` (lr `smulm` (x `vxv` (deriv `vmulv` bp2 ) ))
        biases2 = biases `vsubv` (lr `smulv` (deriv `vmulv` bp2))
        bp3 = ((transp weights) `mmulv` (deriv `vmulv` bp2))
        (VectI bp4) = bp3
    ((SGDLayer numInputs weights2 biases2 lspec), bp4)
        

lspecGetNumOutputs :: LSpec -> Int
lspecGetNumOutputs [] = 0
lspecGetNumOutputs ((Sigmoid x) : rest) = x + (lspecGetNumOutputs rest)

layerGetNumInputs :: Layer -> Int
layerGetNumInputs (SGDLayer num _ _ _ ) = num
