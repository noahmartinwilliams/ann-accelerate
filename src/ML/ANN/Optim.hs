{-# LANGUAGE DeriveGeneric #-}
module ML.ANN.Optim(Optim(..)) where

import Data.Array.Accelerate as A
import Prelude as P
import Data.Serialize

data Optim = SGD Double |  -- learnRate
    Mom Double Double  | -- learnRate Momentum
    RMSProp Double Double | -- alpha beta
    Adagrad Double |
    Adam Double Double Double -- alpha beta1 beta2
    deriving(Show, P.Eq, Read, Generic)

instance Serialize Optim
