{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators, FlexibleContexts #-}
module ML.ANN.Network (Network(..), LNetwork(..), calcNetwork, mkNetwork, learnNetwork, backpropNetwork) where

import Data.Array.Accelerate as A
import Prelude as P
import System.Random
import Data.Random.Normal

import ML.ANN.Layer
import ML.ANN.Optim
import ML.ANN.ActFuncs

data Network = SGDNetwork [Layer] Optim deriving(Show)
data LNetwork = LSGDNetwork [LLayer] Optim deriving(Show)

mkNetwork :: StdGen -> [LSpec] -> Optim -> Network
mkNetwork g lspec (SGD lr) = do
    let rands = normals g
        numInputs = lspecGetNumOutputs (lspec P.!! 0)
    SGDNetwork (internSGD rands lspec numInputs numInputs) (SGD lr) where
        internSGD :: [Double] -> [LSpec] -> Int -> Int -> [Layer]
        internSGD _ [] _ _ = []
        internSGD rands [lspec2] numInputs numOutputs = do
            [(mkSGDLayer rands lspec2 numInputs numOutputs)]
        internSGD rands (lspec2 : rest) numInputs numOutputs = do
            (mkSGDLayer rands lspec2 numInputs numOutputs) : (internSGD rands rest numOutputs (lspecGetNumOutputs (rest P.!! 0)))
                
    
calcNetwork :: Network -> Acc (Vector Double) -> Acc (Vector Double)
calcNetwork (SGDNetwork layers _) x = do
    let x2 = A.replicate (constant (Z:.All:.(1::Int))) x
    A.flatten (calcIntern layers x2) where
        calcIntern :: [Layer] -> Acc (Matrix Double) -> Acc (Matrix Double)
        calcIntern [] y = y
        calcIntern (layer : rest) y = calcIntern rest (calcLayer layer y)


learnNetwork :: Network -> Acc (Vector Double) -> (LNetwork, Acc (Vector Double))
learnNetwork (SGDNetwork layers lr) input = do
    let x = A.replicate (A.lift (Z:.All:.(1::Int))) input
        (llayers, output) = intern layers x
    (LSGDNetwork llayers lr, A.flatten output) where
        intern :: [Layer] -> Acc (Matrix Double) -> ([LLayer], Acc (Matrix Double))
        intern [] input2 = ([], input2)
        intern (h : t) input2 = do
            let (llayer, output) =  learnLayer h input2
                (restllayer, output2) = intern t output
            (llayer : restllayer, output2)

backpropNetwork :: LNetwork -> Acc (Vector Double) -> (Network, Acc (Vector Double))
backpropNetwork (LSGDNetwork llayers lr) backProp = do
    let backProp2 = A.replicate (A.lift (Z:.All:.(1::Int))) backProp
        (layers, retbp) = intern llayers lr backProp2
    ((SGDNetwork layers lr), (A.flatten retbp)) where
        
        intern :: [LLayer] -> Optim -> Acc (Matrix Double) -> ([Layer], Acc (Matrix Double))
        intern [] _ x = ([], x)
        intern ( h : t ) optim bp = do
            let (restLayer, bp2) = intern t optim bp
                (layer, bp3) = backpropLayer h optim bp2
            (layer : restLayer, bp3)
