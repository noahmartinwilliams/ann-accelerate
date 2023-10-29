module ML.ANN.ActFuncs (sigmoid) where

import Data.Array.Accelerate as A
import Prelude as P

sigmoid :: Exp Double -> Exp Double
sigmoid x = let one = constant 1.0 in one / (one + (exp (-x)))
