{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators #-}
module ML.ANN.Network (Network(..), calcNetwork, mkNetwork) where

import Data.Array.Accelerate as A
import Prelude as P
import System.Random
import Data.Random.Normal

import ML.ANN.Layer
import ML.ANN.Optim

data Network = SGDNetwork [Layer] (Exp Double) deriving(Show)

mkNetwork :: StdGen -> [LSpec] -> Optim -> Network
mkNetwork g lspec (SGD lr) = do
    let rands = normals g
        numInputs = lspecGetNumOutputs (lspec P.!! 0)
    SGDNetwork (internSGD rands lspec numInputs numInputs) lr where
        internSGD :: [Double] -> [LSpec] -> Int -> Int -> [Layer]
        internSGD _ [] _ _ = []
        internSGD randoms [lspec] numInputs numOutputs = do
            [(mkSGDLayer randoms lspec numInputs numOutputs)]
        internSGD randoms (lspec : rest) numInputs numOutputs = do
            (mkSGDLayer randoms lspec numInputs numOutputs) : (internSGD randoms rest numOutputs (lspecGetNumOutputs (rest P.!! 0)))
                
    
calcNetwork :: Network -> Acc (Vector Double) -> Acc (Vector Double)
calcNetwork (SGDNetwork layers _) x = do
    let x2 = A.replicate (constant (Z:.All:.(1::Int))) x
    A.flatten (calcIntern layers x2) where
        calcIntern :: [Layer] -> Acc (Matrix Double) -> Acc (Matrix Double)
        calcIntern [] y = y
        calcIntern (layer : rest) y = calcIntern rest (calcLayer layer y)

