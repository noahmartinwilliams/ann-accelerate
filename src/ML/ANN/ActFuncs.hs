{-# LANGUAGE GADTs, TypeFamilies, DataKinds, DeriveGeneric, FlexibleContexts #-}
module ML.ANN.ActFuncs (sigmoid, dsigmoid, relu, drelu, ident, dident, softmax, dsoftmax, _tanh, dtanh, ActFunc(..), applyActFuncs, dapplyActFuncs, getInt) where

import Data.Array.Accelerate as A
import Prelude as P
import ML.ANN.Vect
import ML.ANN.Mat

import Data.Serialize

data ActFunc = Sigmoid Int | Relu Int | Ident Int | Softmax Int | TanH Int deriving(Show, P.Eq, P.Read, Generic)

instance Serialize ActFunc

getInt :: ActFunc -> Int
getInt (Sigmoid i) = i
getInt (Relu i) = i
getInt (Ident i) = i
getInt (Softmax i) = i
getInt (TanH i) = i

sigmoid :: (Acc (Matrix Double) -> Acc (Matrix Double))
sigmoid = let func = (\x -> let one = constant 1.0 in one / (one + (exp (-x)))) in A.map func

dsigmoid :: (Acc (Matrix Double) -> Acc (Matrix Double))
dsigmoid = let func = (\x -> let one = constant 1.0 in let e = exp (-x) in e / ((e + one) * (e + one))) in A.map func 

relu :: (Acc (Matrix Double) -> Acc (Matrix Double))
relu = A.map (\y -> A.max (constant 0.0) y)

drelu :: (Acc (Matrix Double) -> Acc (Matrix Double))
drelu = A.map (\y -> A.fromIntegral (boolToInt (y A.>= (constant 0.0 :: Exp Double))) :: Exp Double)

ident :: (Acc (Matrix Double) -> Acc (Matrix Double))
ident = let f x = x in f

dident :: (Acc (Matrix Double) -> Acc (Matrix Double))
dident = let f x = A.map (\_ -> constant 1.0) x :: Acc (Matrix Double) in f

softmax :: (Acc (Matrix Double) -> Acc (Matrix Double))
softmax = do
    let summation x = A.the (A.sum (A.flatten (A.map (\y -> exp y) x)))
        ret x = let s = summation x in A.map (\y -> (exp y) / s) x
    ret

dsoftmax :: (Acc (Matrix Double) -> Acc (Matrix Double))
dsoftmax = do
       let fn x = A.zipWith (-) (softmax x) (A.zipWith (*) (softmax x) (softmax x))
       fn 

_tanh :: (Acc (Matrix Double) -> Acc (Matrix Double))
_tanh = A.map (\x -> tanh x) 

dtanh :: (Acc (Matrix Double) -> Acc (Matrix Double))
dtanh = A.map (\x -> (sech x) * (sech x)) where
    sech :: Exp Double -> Exp Double
    sech x = (constant 1.0 ) / (cosh x)

getFunc :: ActFunc -> (Acc (Matrix Double) -> Acc (Matrix Double))
getFunc (Sigmoid _) = sigmoid
getFunc (Relu _) = relu
getFunc (Ident _) = ident
getFunc (Softmax _) = softmax
getFunc (TanH _) = _tanh

dgetFunc :: ActFunc -> (Acc (Matrix Double) -> Acc (Matrix Double))
dgetFunc (Sigmoid _) = dsigmoid
dgetFunc (Relu _) = drelu
dgetFunc (Ident _) = dident
dgetFunc (Softmax _) = dsoftmax
dgetFunc (TanH _) = dtanh

applyActFuncs :: [ActFunc] -> Vect OutputSize -> Vect OutputSize
applyActFuncs [] x = x
applyActFuncs ( afunc : rest ) x = do
    let i = getInt afunc
        piece = takeV x (constant i)
        result = (getFunc afunc) piece
        restV = dropV x (constant i)
        (VectO rest2) = applyActFuncs rest (VectO (A.transpose restV))
        rest3 = A.transpose rest2
    VectO (A.transpose (result A.++ rest3))

dapplyActFuncs :: [ActFunc] -> Vect OutputSize -> Vect OutputSize
dapplyActFuncs [] x = x
dapplyActFuncs ( afunc : rest ) x = do
    let i = getInt afunc
        piece = takeV x (constant i)
        result = (dgetFunc afunc) piece
        restV = dropV x (constant i)
        (VectO rest2) = dapplyActFuncs rest (VectO (A.transpose restV))
        rest3 = A.transpose rest2
    VectO (A.transpose (result A.++ rest3))
