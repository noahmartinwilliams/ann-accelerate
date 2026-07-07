{-# LANGUAGE TypeSynonymInstances, FlexibleInstances #-}
module ML.ANN.BString where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter -- If this library is used right then we hopefully won't need a run function more advanced than what the interpreter gives us.
import Data.Array.Accelerate.Matrix
import Data.Binary
import Data.ByteString as B
import Debug.Trace
import ML.ANN.Types
import Prelude as P

dList2bs :: [Double] -> Put
dList2bs [] = return ()
dList2bs (a : r) = do
    put a
    dList2bs r

weights2bs :: (Weights, Int, Int) -> Put
weights2bs ((AccMat m _ _), numIns, numOuts) = do
    put numIns
    put numOuts
    dList2bs (A.toList (run m))

bs2dList :: Int -> Get [Double]
bs2dList 0 = return []
bs2dList i = do
    bs <- get 
    r <- bs2dList (i - 1)
    return (bs : r)

bs2weights :: Get (Weights, Int, Int)
bs2weights = do
    numIns <- get 
    numOuts <- get
    dList <- bs2dList (numIns * numOuts)
    let m = A.fromList (Z:.numOuts:.numIns) dList
    return ((AccMat (use m) Outp Inp), numIns, numOuts)

newtype BW = BW (Weights, Int, Int) --deriving(Generic)
instance Binary BW where
    put (BW w) = weights2bs w
    get = do
        g <- bs2weights
        return (BW g)
