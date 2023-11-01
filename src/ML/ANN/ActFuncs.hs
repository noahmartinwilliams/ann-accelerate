module ML.ANN.ActFuncs (sigmoid, ActFunc(..)) where

import Data.Array.Accelerate as A
import Prelude as P

data ActFunc = Sigmoid Int deriving(Show, Eq)

sigmoid :: Exp Double -> Exp Double
sigmoid x = let one = constant 1.0 in one / (one + (exp (-x)))
