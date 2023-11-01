module ML.ANN.Layer (Layer(..), LSpec()) where

import Data.Array.Accelerate as A
import Prelude as P

import ML.ANN.Mat 
import ML.ANN.Vect
import ML.ANN.ActFuncs

type LSpec = [ActFunc]

data Layer = SGDLayer (Mat OutputSize InputSize) (Vect OutputSize) LSpec
