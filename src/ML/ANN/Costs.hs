module ML.ANN.Costs(CostFnT(..), costFn, CostFn, dmseFn, mseFn, mseCFn, dcrossEntropyFn, crossEntropyFn, crossEntropyCFn) where

import Data.Array.Accelerate as A
import Prelude as P

data CostFnT = MSE | CrossEntropy deriving(Show, Read)

costFn :: CostFnT -> CostFn
costFn MSE = mseCFn
costFn CrossEntropy = crossEntropyCFn

type CostFn = ((Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)), (Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)))

dmseFn :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
dmseFn correct actual = do
    let len = A.fromIntegral (A.length actual) :: Exp Double
    A.zipWith (\x -> \y -> - (constant 2.0) * ( x - y ) / len) correct actual

mseFn :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
mseFn correct actual = do
    let len = A.fromIntegral (A.length actual) :: Exp Double
    A.zipWith (\x -> \y -> ( x - y ) * ( x - y ) / len) correct actual

mseCFn :: CostFn
mseCFn = (mseFn, dmseFn)

dcrossEntropyFn :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
dcrossEntropyFn correct actual = do
    let one = constant 1.0  
        l = A.fromIntegral (A.length correct) :: Exp Double
        dfn x y = (((one - x) / (one - y)) - (x/y)) / l
    A.zipWith (\x -> \y -> dfn x y ) correct actual

crossEntropyFn :: Acc (Vector Double) -> Acc (Vector Double) -> Acc (Vector Double)
crossEntropyFn correct actual = do
    let one = constant 1.0 
        l = A.fromIntegral (A.length correct) :: Exp Double
    A.zipWith (\x -> \y -> -(x * (log y) + (one - x) * (log (one - y))) / l) correct actual

crossEntropyCFn :: CostFn
crossEntropyCFn = (dcrossEntropyFn, crossEntropyFn)
