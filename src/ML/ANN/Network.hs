{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators, FlexibleContexts #-}
module ML.ANN.Network (Network(..), LNetwork(..), calcNetwork, mkNetwork, learnNetwork, backpropNetwork) where

import Data.Array.Accelerate as A
import Prelude as P
import System.Random
import Data.Random.Normal

import ML.ANN.Layer
import ML.ANN.Optim
import ML.ANN.ActFuncs

data Network = Network [Layer] Optim deriving(Show)

data LNetwork = LNetwork [LLayer] Optim deriving(Show)

mkNetwork :: StdGen -> [LSpec] -> Optim -> Network
mkNetwork g lspec (RMSProp alpha beta) | (P.length lspec) P.>=2 = do
    let rands = normals g
        numInputs = lspecGetNumOutputs (lspec P.!! 0)
        (firstLayer : restLayers) = lspec
        layer1 = mkRMSInpLayer rands firstLayer numInputs
        numOutputs = lspecGetNumOutputs (lspec P.!! 1)
    Network (layer1 : (internRMS rands restLayers numInputs numOutputs)) (RMSProp alpha beta) where
        internRMS :: [Double] -> [LSpec] -> Int -> Int -> [Layer]
        internRMS _ [] _ _ = []
        internRMS rands [lspec2] numInputs numOutputs = do
            [(mkRMSLayer rands lspec2 numInputs numOutputs)]
        internRMS rands (lspec2 : rest) numInputs numOutputs = do
            (mkRMSLayer rands lspec2 numInputs numOutputs) : (internRMS rands rest numOutputs (lspecGetNumOutputs (rest P.!! 0)))

mkNetwork g lspec (Mom alpha beta) | (P.length lspec) P.>=2 = do
    let rands = normals g
        numInputs = lspecGetNumOutputs (lspec P.!! 0)
        (firstLayer : restLayers) = lspec
        layer1 = mkMomInpLayer rands firstLayer numInputs
        numOutputs = lspecGetNumOutputs (lspec P.!! 1)
    Network (layer1 : (internMom rands restLayers numInputs numOutputs)) (Mom alpha beta) where
        internMom :: [Double] -> [LSpec] -> Int -> Int -> [Layer]
        internMom _ [] _ _ = []
        internMom rands [lspec2] numInputs numOutputs = do
            [(mkMomLayer rands lspec2 numInputs numOutputs)]
        internMom rands (lspec2 : rest) numInputs numOutputs = do
            (mkMomLayer rands lspec2 numInputs numOutputs) : (internMom rands rest numOutputs (lspecGetNumOutputs (rest P.!! 0)))

mkNetwork g lspec (SGD lr) | (P.length lspec) P.>= 2 = do
    let rands = normals g
        numInputs = lspecGetNumOutputs (lspec P.!! 0)
        (firstLayer : restLayers) = lspec
        layer1 = mkSGDInpLayer rands firstLayer
        numOutputs = lspecGetNumOutputs (lspec P.!! 1)
    Network (layer1 : (internSGD rands restLayers numInputs numOutputs)) (SGD lr) where
        internSGD :: [Double] -> [LSpec] -> Int -> Int -> [Layer]
        internSGD _ [] _ _ = []
        internSGD rands [lspec2] numInputs numOutputs = do
            [(mkSGDLayer rands lspec2 numInputs numOutputs)]
        internSGD rands (lspec2 : rest) numInputs numOutputs = do
            (mkSGDLayer rands lspec2 numInputs numOutputs) : (internSGD rands rest numOutputs (lspecGetNumOutputs (rest P.!! 0)))

                
    
calcNetwork :: Network -> Acc (Vector Double) -> Acc (Vector Double)
calcNetwork (Network layers _) x = do
    let x2 = A.replicate (constant (Z:.All:.(1::Int))) x
    A.flatten (calcIntern layers x2) where
        calcIntern :: [Layer] -> Acc (Matrix Double) -> Acc (Matrix Double)
        calcIntern [] y = y
        calcIntern (layer : rest) y = calcIntern rest (calcLayer layer y)


learnNetwork :: Network -> Acc (Vector Double) -> (LNetwork, Acc (Vector Double))
learnNetwork (Network layers lr) input = do
    let x = A.replicate (A.lift (Z:.All:.(1::Int))) input
        (llayers, output) = intern layers x
    (LNetwork llayers lr, A.flatten output) where
        intern :: [Layer] -> Acc (Matrix Double) -> ([LLayer], Acc (Matrix Double))
        intern [] input2 = ([], input2)
        intern (h : t) input2 = do
            let (llayer, output) =  learnLayer h input2
                (restllayer, output2) = intern t output
            (llayer : restllayer, output2)

backpropNetwork :: LNetwork -> Acc (Vector Double) -> (Network, Acc (Vector Double))
backpropNetwork (LNetwork llayers lr) backProp = do
    let backProp2 = A.replicate (A.lift (Z:.All:.(1::Int))) backProp
        (layers, retbp) = intern llayers lr backProp2
    ((Network layers lr), (A.flatten retbp)) where
        
        intern :: [LLayer] -> Optim -> Acc (Matrix Double) -> ([Layer], Acc (Matrix Double))
        intern [] _ x = ([], x)
        intern ( h : t ) optim bp = do
            let (restLayer, bp2) = intern t optim bp
                (layer, bp3) = backpropLayer h optim bp2
            (layer : restLayer, bp3)

