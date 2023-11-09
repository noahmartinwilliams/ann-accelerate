module ML.ANN.Optim(Optim(..)) where

import Data.Array.Accelerate as A
import Prelude as P

data Optim = SGD (Exp Double) deriving(Show)
