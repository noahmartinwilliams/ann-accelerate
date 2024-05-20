{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators, FlexibleContexts #-}
module ML.ANN
    ( module ML.ANN.Costs,
    module ML.ANN.Network,
    module ML.ANN.Optim,
    sigmoid,
    dsigmoid,
    relu,
    drelu,
    ident,
    dident,
    softmax,
    dsoftmax,
    ActFunc(..),
    module ML.ANN.Block,
    module ML.ANN.File,
    ANN(..),
    trainOnce,
    ML.ANN.Layer.LSpec(..),
    normalize,
    lspecGetNumOutputs,
    lspecGetNumInputs
    ) where

import ML.ANN.Costs
import ML.ANN.Network
import ML.ANN.Optim
import ML.ANN.ActFuncs
import ML.ANN.Block
import ML.ANN.File
import ML.ANN.Layer
import Data.Array.Accelerate as A

data ANN = ANN BlockInfo BlockA deriving(Show)

trainOnce :: ANN -> CostFn -> Acc (Vector Double, Vector Double) -> Acc (Vector Double, Vector Int, Vector Double)
trainOnce (ANN blinfo block) (costFnErr, costFnDeriv) sample = do
    let net = block2network (blinfo, block)
        (input, output) = A.unlift sample
        (ln, actual) = learnNetwork net input
        (net2, _) = backpropNetwork ln (costFnDeriv output actual)
        err = costFnErr output actual
        (_, blockOut) = network2block net2
        (blockOutI, blockOutD) = A.unlift blockOut :: (Acc (Vector Int), Acc (Vector Double))
        output2 = A.lift (err, blockOutI, blockOutD)
    output2

normalize :: (Shape sh) => Acc (Array sh Double) -> Acc (Array sh Double)
normalize input = do
    let squared = A.map (\x -> x * x) input
        summed = A.the (A.map (\x -> sqrt x) (A.sum (A.flatten squared)))
    A.map (\x -> x / summed) input
