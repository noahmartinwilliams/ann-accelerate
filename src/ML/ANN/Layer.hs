{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators #-}
module ML.ANN.Layer (Layer(..), LLayer(..), LSpec(), calcLayer, lspecGetNumOutputs, layerGetNumInputs, mkSGDLayer, mkSGDInpLayer, learnLayer, backpropLayer) where

import Data.Array.Accelerate as A
import Prelude as P

import ML.ANN.Mat 
import ML.ANN.Vect
import ML.ANN.ActFuncs
import ML.ANN.Optim
import System.Random
import System.IO
import Data.Random.Normal

type LSpec = [ActFunc]

data Layer = SGDLayer Int (Mat OutputSize InputSize) (Vect OutputSize) LSpec | -- NumInputs weights bias lspec
    SGDInpLayer (Vect OutputSize) (Vect OutputSize) LSpec | -- weights bias lspec
    deriving(Show) 

data LLayer = LSGDLayer Layer (Vect InputSize) | -- layer previousInput
    LSGDInpLayer Layer (Vect OutputSize) deriving(Show)

mkSGDInpLayer :: [Double] -> LSpec -> Layer
mkSGDInpLayer randoms lspec = do
    let numInputs = lspecGetNumOutputs lspec
        randoms2 = P.map (\x -> x * (sqrt (2.0 / (P.fromIntegral numInputs :: Double)))) randoms
        weights = VectO (use (fromList (Z:.numInputs:.1) (P.take numInputs randoms2)))
        biases = weights
    SGDInpLayer weights biases lspec


mkSGDLayer :: [Double] -> LSpec -> Int -> Int -> Layer
mkSGDLayer randoms lspec numInputs numOutputs = do
    let randoms2 = P.map (\x -> x * (sqrt (2.0 / (P.fromIntegral numInputs :: Double)))) randoms :: [Double]
    let weightsM = use (fromList (Z:.numOutputs:.numInputs) randoms2)
        biasesM = use (fromList (Z:.numOutputs:.1) randoms2)
    SGDLayer numInputs (MatOI weightsM) (VectO biasesM) lspec

calcLayer :: Layer -> Acc (Matrix Double) -> Acc (Matrix Double)
calcLayer (SGDInpLayer weights bias lspec) x = do
    let x2 = (weights `vmulv` (VectO x)) `vaddv` bias
        (VectO output) = applyActFuncs lspec x2
    output
calcLayer (SGDLayer _ weights bias lspec) x = do
    let x2 = VectI x
        (VectO output) = applyActFuncs lspec ((weights `mmulv` x2 ) `vaddv` bias)
    output

learnLayer :: Layer -> Acc (Matrix Double) -> (LLayer, Acc (Matrix Double))
learnLayer (SGDInpLayer weights biases lspec) input = do
    let x = VectO input
        output = applyActFuncs lspec ((weights `vmulv` x) `vaddv` biases)
    ((LSGDInpLayer (SGDInpLayer weights biases lspec) x), (extractVect output))

learnLayer (SGDLayer numInputs weights biases lspec) input = do
    let x = VectI input
        output = applyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
    ((LSGDLayer (SGDLayer numInputs weights biases lspec) x), (extractVect output))

backpropLayer :: LLayer -> Optim -> Acc (Matrix Double) -> (Layer, Acc (Matrix Double))
backpropLayer (LSGDInpLayer layer x) (SGD learnRate) bp = do
    let lr = constant learnRate
        (SGDInpLayer weights biases lspec) = layer
        bp2 = VectO bp
        deriv = dapplyActFuncs lspec ((weights `vmulv` x) `vaddv` biases)
        weights2 = weights `vsubv` (lr `smulv` (x `vmulv` (deriv `vmulv` bp2)))
        biases2 = biases `vsubv` (lr `smulv` (deriv `vmulv` bp2))
        bp3 = weights `vmulv` (deriv `vmulv` bp2)
        (VectO bp4) = bp3
    ((SGDInpLayer weights2 biases2 lspec), bp4)
backpropLayer (LSGDLayer layer x) (SGD learnRate) bp = do
    let lr = constant learnRate
        (SGDLayer numInputs weights biases lspec) = layer
        bp2 = VectO bp
        deriv = dapplyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
        weights2 = weights `msubm` (lr `smulm` (x `vxv` (deriv `vmulv` bp2 ) ))
        biases2 = biases `vsubv` (lr `smulv` (deriv `vmulv` bp2))
        bp3 = ((transp weights) `mmulv`  (deriv `vmulv` bp2))
        (VectI bp4) = bp3
    ((SGDLayer numInputs weights2 biases2 lspec), bp4)
        

lspecGetNumOutputs :: LSpec -> Int
lspecGetNumOutputs [] = 0
lspecGetNumOutputs ((Sigmoid x) : rest) = x + (lspecGetNumOutputs rest)
lspecGetNumOutputs ((Relu x) : rest) = x + (lspecGetNumOutputs rest)

layerGetNumInputs :: Layer -> Int
layerGetNumInputs (SGDLayer num _ _ _ ) = num
