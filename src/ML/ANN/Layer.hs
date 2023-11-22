{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators #-}
module ML.ANN.Layer (Layer(..), LLayer(..), LSpec(), calcLayer, lspecGetNumOutputs, layerGetNumInputs, mkSGDLayer, mkSGDInpLayer, learnLayer, backpropLayer, mkMomLayer) where

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
    MomLayer Int (Mat OutputSize InputSize) (Mat OutputSize InputSize) (Vect OutputSize) (Vect OutputSize) LSpec -- numInputs weights weightsMomentum biases biasesMomentum lspec
    deriving(Show) 

data LLayer = LSGDLayer Layer (Vect InputSize) | -- layer previousInput
    LSGDInpLayer Layer (Vect OutputSize) | -- layer previousInput
    LMomLayer Layer (Vect InputSize) deriving(Show)

mkSGDInpLayer :: [Double] -> LSpec -> Layer
mkSGDInpLayer randoms lspec = do
    let numInputs = lspecGetNumOutputs lspec
        randoms2 = P.map (\x -> x * (sqrt (2.0 / (P.fromIntegral numInputs :: Double)))) randoms
        weights = VectO (use (fromList (Z:.numInputs:.1) (P.take numInputs randoms2)))
        biases = weights
    SGDInpLayer weights biases lspec

heWeightInit :: [Double] -> Int -> [Double]
heWeightInit randoms numInputs = P.map (\x -> x * (sqrt (2.0 / (P.fromIntegral numInputs :: Double)))) randoms

mkSGDLayer :: [Double] -> LSpec -> Int -> Int -> Layer
mkSGDLayer randoms lspec numInputs numOutputs = do
    let randoms2 = heWeightInit randoms numInputs
        weightsM = use (fromList (Z:.numOutputs:.numInputs) randoms2)
        biasesM = use (fromList (Z:.numOutputs:.1) randoms2)
    SGDLayer numInputs (MatOI weightsM) (VectO biasesM) lspec

mkMomLayer :: [Double] -> LSpec -> Int -> Int -> Layer
mkMomLayer randoms lspec numInputs numOutputs = do
    let randoms2 = heWeightInit randoms numInputs
        weightsM = use (fromList (Z:.numOutputs:.numInputs) randoms2)
        biasesM = use (fromList (Z:.numOutputs:.1) randoms2)
        weightsMom = use (fromList (Z:.numOutputs:.numInputs) (P.repeat 0.0))
        biasesMom = use (fromList (Z:.numOutputs:.1) (P.repeat 0.0))
    MomLayer numInputs (MatOI weightsM) (MatOI weightsMom) (VectO biasesM) (VectO biasesMom) lspec


calcLayer :: Layer -> Acc (Matrix Double) -> Acc (Matrix Double)
calcLayer (SGDInpLayer weights bias lspec) x = do
    let x2 = (weights `vmulv` (VectO x)) `vaddv` bias
        (VectO output) = applyActFuncs lspec x2
    output
calcLayer (SGDLayer _ weights bias lspec) x = do
    let x2 = VectI x
        (VectO output) = applyActFuncs lspec ((weights `mmulv` x2 ) `vaddv` bias)
    output
calcLayer (MomLayer _ weights _ biases _ lspec) x = do
    let x2 = VectI x
        (VectO output) = applyActFuncs lspec ((weights `mmulv` x2) `vaddv` biases)
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

learnLayer (MomLayer numInputs weights weightsMom biases biasesMom lspec) input = do
    let x = VectI input
        output = applyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
    ((LMomLayer (MomLayer numInputs weights weightsMom biases biasesMom lspec) x), (extractVect output))


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

backpropLayer (LMomLayer (MomLayer numInputs weights weightsMom biases biasesMom lspec) x) (Mom alpha mom) bpInp = do
    let deriv = dapplyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
        bp = VectO bpInp
        momExp = constant mom
        alphaExp = constant alpha
        fprimeW = x `vxv` (bp `vmulv` deriv)
        fprimeB = bp `vmulv` deriv
        changeWeights = (alphaExp `smulm` fprimeW) `maddm` (momExp `smulm` weightsMom)
        changeBiases = (alphaExp `smulv` fprimeB) `vaddv` (momExp `smulv` biasesMom)
        --weightsMom2 = weightsMom `maddm` (momExp `smulm` fprimeW)
        --biasesMom2 = biasesMom `vaddv` (momExp `smulm` fprimeB)
        weights2 = weights `msubm` changeWeights
        biases2 = biases `vsubv` changeBiases
        bp2 = (transp weights) `mmulv` (bp `vmulv` deriv)
        (VectI bp3) = bp2
    ((MomLayer numInputs weights2 changeWeights biases changeBiases lspec), bp3)


lspecGetNumOutputs :: LSpec -> Int
lspecGetNumOutputs [] = 0
lspecGetNumOutputs (h : t) = (getInt h) + (lspecGetNumOutputs t)

layerGetNumInputs :: Layer -> Int
layerGetNumInputs (SGDLayer num _ _ _ ) = num
layerGetNumInputs (MomLayer num _ _ _ _ _) = num
