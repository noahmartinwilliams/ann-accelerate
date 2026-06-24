module ML.ANN.ActFunc where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.ANN.Types
import Prelude as P

sigmoid :: Exp Double -> Exp Double
sigmoid x = let one = constant 1.0 in one / (one + (exp (-x)))

dsigmoid :: Exp Double -> Exp Double
dsigmoid x = do
    let one = constant 1.0
        e = exp (-x)
        es = (e + one) * (e + one)
    e / es

relu :: Exp Double -> Exp Double
relu x = A.max (constant 0.0) x

softmax :: Acc (Matrix Double) -> Acc (Matrix Double)
softmax x = do
    let e = A.map (\y -> exp y) x
        s = A.sum (A.flatten e)
        s' = A.the s
    A.map (\y -> y / s') e

data IntermSlice = IntermSlice
data IntermRest = IntermRest

actFunc :: ActFunc -> AccMat Double a b -> AccMat Double a b
actFunc Sigmoid (AccMat inp a b) = AccMat (A.map sigmoid inp) a b
actFunc Relu (AccMat inp a b) = AccMat (A.map relu inp) a b
actFunc SoftMax (AccMat inp a b) = AccMat (softmax inp) a b

actFuncs' :: LSpec -> AccMat Double One IntermRest -> AccMat Double One IntermSlice
actFuncs' [] (AccMat m a b) = AccMat m a IntermSlice
actFuncs' ((i, af) : rest) am = do
    let m' = matTake (constant i) am IntermSlice
        m'' = actFunc af m'
        amrest = matDrop (constant i) am IntermRest
        m2 = actFuncs' rest amrest 
    (matAppend m'' m2 IntermSlice)

actFuncs :: LSpec -> AccMat Double Outp One -> AccMat Double Outp One
actFuncs lspec m = do
    let (AccMat mIntern One Outp) = matTransp m
        m' = actFuncs' lspec (AccMat mIntern One IntermRest) 
        (AccMat mIntern' IntermSlice One) = matTransp m'
    (AccMat mIntern' Outp One)

dactFunc :: ActFunc -> AccMat Double One a -> AccMat Double One a
dactFunc Sigmoid (AccMat m One a) = AccMat (A.map dsigmoid m) One a
dactFunc Relu (AccMat m One a) = AccMat (A.map (\x -> (x A.>= (constant 0.0)) A.? (constant 1.0, constant 0.0)) m) One a
dactFunc SoftMax (AccMat m One a) = do
    let s = softmax m
        s' = A.zipWith (-) s (A.map (\x -> x * x) s)
    AccMat (softmax s') One a


dactFuncs' :: LSpec -> AccMat Double One IntermRest -> AccMat Double One IntermSlice
dactFuncs' [] (AccMat m One IntermRest) = AccMat m One IntermSlice
dactFuncs' ((i, af) : rest) am = do
    let m' = matTake (constant i) am IntermSlice
        m'' = dactFunc af m'
        amrest = matDrop (constant i) am IntermRest
        m2 = dactFuncs' rest amrest 
    (matAppend m'' m2 IntermSlice)
    
dactFuncs :: LSpec -> AccMat Double Outp One -> AccMat Double Outp One
dactFuncs lspec m = do
    let m' = matTransp m
        (AccMat am One Outp) = m'
        (AccMat am' One IntermSlice) = dactFuncs' lspec (AccMat am One IntermRest)
    matTransp (AccMat am' One Outp)
