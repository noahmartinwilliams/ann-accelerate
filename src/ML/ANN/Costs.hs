module ML.ANN.Costs(dlinRegCostFn, linRegCostFn) where

import Data.Array.Accelerate as A
import Prelude as P

dlinRegCostFn :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
dlinRegCostFn correct actual = A.zipWith (\x -> \y -> - (constant 2.0) * ( x - y ) ) correct actual

linRegCostFn :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
linRegCostFn correct actual = A.zipWith (\x -> \y -> ( x - y ) * ( x - y )) correct actual
