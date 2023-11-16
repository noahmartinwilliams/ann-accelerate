{-# LANGUAGE GADTs, TypeFamilies, DataKinds, DeriveGeneric #-}
module ML.ANN.ActFuncs (sigmoid, dsigmoid, ActFunc(..), applyActFuncs, dapplyActFuncs) where

import Data.Array.Accelerate as A
import Prelude as P
import ML.ANN.Vect
import ML.ANN.Mat

import Data.Serialize

data ActFunc = Sigmoid Int | Relu Int deriving(Show, P.Eq, P.Read, Generic)

instance Serialize ActFunc

getInt :: ActFunc -> Int -- TODO: Find a better way to do this.
getInt (Sigmoid i) = i
getInt (Relu i) = i

sigmoid :: (Acc (Matrix Double) -> Acc (Matrix Double))
sigmoid = let func = (\x -> let one = constant 1.0 in one / (one + (exp (-x)))) in A.map func

dsigmoid :: (Acc (Matrix Double) -> Acc (Matrix Double))
dsigmoid = let func = (\x -> let one = constant 1.0 in let e = exp (-x) in e / ((e + one) * (e + one))) in A.map func 

relu :: (Acc (Matrix Double) -> Acc (Matrix Double))
relu = A.map (\y -> A.max (constant 0.0) y)

drelu :: (Acc (Matrix Double) -> Acc (Matrix Double))
drelu = A.map (\y -> A.fromIntegral (boolToInt (y A.>= (constant 0.0 :: Exp Double))) :: Exp Double)

getFunc :: ActFunc -> (Acc (Matrix Double) -> Acc (Matrix Double))
getFunc (Sigmoid _) = sigmoid
getFunc (Relu _) = relu

dgetFunc :: ActFunc -> (Acc (Matrix Double) -> Acc (Matrix Double))
dgetFunc (Sigmoid _) = dsigmoid
dgetFunc (Relu _) = drelu

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
