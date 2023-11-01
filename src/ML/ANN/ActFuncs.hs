{-# LANGUAGE GADTs, TypeFamilies, DataKinds #-}
module ML.ANN.ActFuncs (sigmoid, ActFunc(..)) where

import Data.Array.Accelerate as A
import Prelude as P
import ML.ANN.Vect
import ML.ANN.Mat

data ActFunc = Sigmoid Int deriving(Show, P.Eq)

sigmoid :: Exp Double -> Exp Double
sigmoid x = let one = constant 1.0 in one / (one + (exp (-x)))

applyActFuncs :: [ActFunc] -> Vect OutputSize -> Vect OutputSize
applyActFuncs [] x = x
applyActFuncs ( ( Sigmoid i ) : rest ) x = do
    let piece = takeV x (constant i)
        result = A.map (sigmoid) piece
        restV = dropV x (constant i)
        (VectO rest2) = applyActFuncs rest (VectO (A.transpose restV))
        rest3 = A.transpose rest2
    VectO (A.transpose (result A.++ rest3))
