{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators, FlexibleContexts #-}
module ML.ANN.Network (Network(..), LNetwork(..), calcNetwork, mkNetwork, learnNetwork, backpropNetwork) where

import Data.Array.Accelerate as A
import Prelude as P
import System.Random
import Data.Random.Normal

import ML.ANN.Layer
import ML.ANN.Optim
import ML.ANN.ActFuncs

data Network = Network [Layer] Optim (Acc (Scalar Int)) deriving(Show)

data LNetwork = LNetwork [LLayer] Optim (Acc (Scalar Int)) deriving(Show)

mkNetwork :: StdGen -> [LSpec] -> Optim -> Network
mkNetwork g lspec (Adam alpha beta1 beta2) | (P.length lspec) P.>=2 = do
    let rands = normals g
        numInputs = lspecGetNumOutputs (lspec P.!! 0)
        (firstLayer : restLayers) = lspec
        layer1 = mkAdamInpLayer rands firstLayer numInputs
        numOutputs = lspecGetNumOutputs (lspec P.!! 1)
    Network (layer1 : (mkNetworkLayers mkAdamLayer rands restLayers numInputs numOutputs)) (Adam alpha beta1 beta2) (use (fromList (Z) [0]))
mkNetwork g lspec (RMSProp alpha beta) | (P.length lspec) P.>=2 = do
    let rands = normals g
        numInputs = lspecGetNumOutputs (lspec P.!! 0)
        (firstLayer : restLayers) = lspec
        layer1 = mkRMSInpLayer rands firstLayer numInputs
        numOutputs = lspecGetNumOutputs (lspec P.!! 1)
    Network (layer1 : (mkNetworkLayers mkRMSLayer rands restLayers numInputs numOutputs)) (RMSProp alpha beta) (use (fromList (Z) [0]))

mkNetwork g lspec (Mom alpha beta) | (P.length lspec) P.>=2 = do
    let rands = normals g
        numInputs = lspecGetNumOutputs (lspec P.!! 0)
        (firstLayer : restLayers) = lspec
        layer1 = mkMomInpLayer rands firstLayer numInputs
        numOutputs = lspecGetNumOutputs (lspec P.!! 1)
    Network (layer1 : (mkNetworkLayers mkMomLayer rands restLayers numInputs numOutputs)) (Mom alpha beta) (use (fromList (Z) [0]))

mkNetwork g lspec (SGD lr) | (P.length lspec) P.>= 2 = do
    let rands = normals g
        numInputs = lspecGetNumOutputs (lspec P.!! 0)
        (firstLayer : restLayers) = lspec
        layer1 = mkSGDInpLayer rands firstLayer
        numOutputs = lspecGetNumOutputs (lspec P.!! 1)
    Network (layer1 : (mkNetworkLayers mkSGDLayer rands restLayers numInputs numOutputs)) (SGD lr) (use (fromList (Z) [0]))
mkNetwork g lspec (Adagrad lr) | (P.length lspec) P.>= 2 = do
    let rands = normals g
        numInputs = lspecGetNumOutputs (lspec P.!! 0)
        (firstLayer : restLayers) = lspec
        layer1 = mkAdagradInpLayer rands firstLayer numInputs
        numOutputs = lspecGetNumOutputs (lspec P.!! 1)
    Network (layer1 : (mkNetworkLayers mkAdagradLayer rands restLayers numInputs numOutputs)) (Adagrad lr) (use (fromList (Z) [0])) 

mkNetworkLayers :: ([Double] -> LSpec -> Int -> Int -> Layer) -> [Double] -> [LSpec] -> Int -> Int -> [Layer]
mkNetworkLayers _ _ [] _ _ = []
mkNetworkLayers fn rands [lspec2] numInputs numOutputs = do
    [(fn rands lspec2 numInputs numOutputs)]
mkNetworkLayers fn rands (lspec2 : rest)  numInputs numOutputs = do
    (fn rands lspec2 numInputs numOutputs) : (mkNetworkLayers fn rands rest numOutputs (lspecGetNumOutputs (rest P.!! 0)))
    
                
    
calcNetwork :: Network -> Acc (Vector Double) -> Acc (Vector Double)
calcNetwork (Network layers _ _) x = do
    let x2 = A.replicate (constant (Z:.All:.(1::Int))) x
    A.flatten (calcIntern layers x2) where
        calcIntern :: [Layer] -> Acc (Matrix Double) -> Acc (Matrix Double)
        calcIntern [] y = y
        calcIntern (layer : rest) y = calcIntern rest (calcLayer layer y)


learnNetwork :: Network -> Acc (Vector Double) -> (LNetwork, Acc (Vector Double))
learnNetwork (Network layers lr numTimes) input = do
    let x = A.replicate (A.lift (Z:.All:.(1::Int))) input
        (llayers, output) = intern layers x
    (LNetwork llayers lr numTimes, A.flatten output) where
        intern :: [Layer] -> Acc (Matrix Double) -> ([LLayer], Acc (Matrix Double))
        intern [] input2 = ([], input2)
        intern (h : t) input2 = do
            let (llayer, output) =  learnLayer h input2
                (restllayer, output2) = intern t output
            (llayer : restllayer, output2)

backpropNetwork :: LNetwork -> Acc (Vector Double) -> (Network, Acc (Vector Double))
backpropNetwork (LNetwork llayers lr numTimes) backProp = do
    let backProp2 = A.replicate (A.lift (Z:.All:.(1::Int))) backProp
        numTimes' = A.map (\x -> x + (constant 1)) numTimes
        (layers, retbp) = intern llayers lr backProp2 numTimes'
    ((Network layers lr numTimes' ), (A.flatten retbp)) where
        
        intern :: [LLayer] -> Optim -> Acc (Matrix Double) -> Acc (Scalar Int) -> ([Layer], Acc (Matrix Double))
        intern [] _ x _ = ([], x)
        intern ( h : t ) optim bp time = do
            let time' = A.the time
                (restLayer, bp2) = intern t optim bp time
                (layer, bp3) = backpropLayer h optim bp2 time'
            (layer : restLayer, bp3)

