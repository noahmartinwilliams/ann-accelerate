{-# LANGUAGE TypeSynonymInstances, FlexibleInstances #-}
module ML.ANN.BString where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter -- If this library is used right then we hopefully won't need a run function more advanced than what the interpreter gives us.
import Data.Array.Accelerate.Matrix
import Data.Binary
import Data.ByteString as B
import Data.Word
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

biases2bs :: Biases -> Put
biases2bs (AccMat biases _ _) = do
    let n = P.length (A.toList (run biases))
    put n 
    dList2bs (A.toList (run biases))

bs2biases :: Get Biases
bs2biases = do
    numBiases <- get 
    b <- bs2dList numBiases
    return (AccMat (use (A.fromList (Z:.numBiases:.1) b)) Outp One)

lspec2bs :: LSpec -> Put
lspec2bs lsp = do
    let n = P.length lsp
    put n
    intern lsp where
        intern :: LSpec -> Put
        intern [] = return ()
        intern ((i, af) : r) = do
            put i
            af2bs af
            intern r

af2bs :: ActFunc -> Put
af2bs Sigmoid = put (0 :: Word8)
af2bs Relu = put (1 :: Word8)
af2bs SoftMax = put (2 :: Word8)

bs2af :: Get ActFunc
bs2af = do
    t <- getWord8 
    case t of
        0 -> return Sigmoid
        1 -> return Relu
        2 -> return SoftMax

bs2lspec :: Get LSpec
bs2lspec = do
    i <- get 
    intern i where
        intern :: Int -> Get LSpec
        intern 0 = return []
        intern i = do
            numNodes <- get 
            af <- bs2af
            r <- intern (i - 1)
            return ((numNodes, af) : r)


edouble2bs :: Exp Double -> Put
edouble2bs d = do
    let d' = (A.toList (run (A.unit d))) P.!! 0
    put d'

eint2bs :: Exp Int -> Put
eint2bs i = do
    let i' = (A.toList (run (A.unit i))) P.!! 0
    put i'

errorFnT2bs :: ErrorFnT -> Put
errorFnT2bs MSEErrorFn = put (0 :: Word8 )
errorFnT2bs CrossEntropyErrorFn = put (1 :: Word8)

bs2errorFn :: Get ErrorFnT
bs2errorFn = do
    t <- getWord8 
    case t of
        0 -> return MSEErrorFn
        1 -> return CrossEntropyErrorFn 

bs2edouble :: Get (Exp Double)
bs2edouble = do
    d <- get
    return (constant d)

bs2eint :: Get (Exp Int)
bs2eint = do
    i <- get
    return (constant i)

optim2bs :: Optim -> Put
optim2bs (SGDOptim lr) = do
    put (0 :: Word8)
    edouble2bs lr
optim2bs (AdamOptim lr b1 b2) = do
    put (1 :: Word8)
    edouble2bs lr
    edouble2bs b1
    edouble2bs b2

bs2optim :: Get Optim
bs2optim = do
    t <- getWord8 
    case t of
        0 -> do
            lr <- bs2edouble
            return (SGDOptim lr)
        1 -> do
            lr <- bs2edouble
            b1 <- bs2edouble
            b2 <- bs2edouble
            return (AdamOptim lr b1 b2)

newtype BW = BW (Weights, Int, Int) 
instance Binary BW where
    put (BW w) = weights2bs w
    get = do
        g <- bs2weights
        return (BW g)

newtype BB = BB Biases
instance Binary BB where
    put (BB b) = biases2bs b
    get = do
        g <- bs2biases 
        return (BB g)

