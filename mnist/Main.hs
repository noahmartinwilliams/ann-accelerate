{-# LANGUAGE BangPatterns #-}
module Main where

import ML.ANN
import Data.Array.Accelerate as A
import Data.Array.Accelerate.LLVM.PTX
import Prelude as P
import System.IO
import System.Random
import Data.ByteString as BS
import Data.List.Split
import Data.Char
import Text.Printf
import System.Console.GetOpt
import System.Environment
import Data.Maybe
import Control.Parallel.Strategies
import GHC.Conc

charToList :: Int -> [Double]
charToList 0 = [0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
charToList 1 = [0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
charToList 2 = [0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
charToList 3 = [0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
charToList 4 = [0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0, 0.0]
charToList 5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0]
charToList 6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0]
charToList 7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0, 0.0]
charToList 8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99, 0.0]
charToList 9 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.99]

getLabels :: BS.ByteString -> [Vector Double]
getLabels bs = do
    let unpacked = BS.unpack bs
        characters = P.map (\x -> P.fromIntegral x :: Int ) unpacked
        lists = P.map (\x -> charToList x) characters
        vects = P.map (\x -> fromList (Z:.10) x) lists
    vects

getImages :: BS.ByteString -> [Vector Double]
getImages bs = do
    let unpacked = BS.unpack bs
        integers = P.map (\x -> P.fromIntegral x :: Int) unpacked
        doubles = P.map (\x -> P.fromIntegral x :: Double) integers
        adjusted = P.map (\x -> (x - 128.0) / 128.0) doubles
        chunked = chunksOf (28*28) adjusted
        vectored = P.map (\x -> fromList (Z:.(28*28)) x) chunked
    vectored

train :: BlockV -> (BlockV -> (Vector Double, Vector Double) -> (Vector Double, Vector Int, Vector Double)) -> [(Vector Double, Vector Double)] -> ([Double], BlockV)
train blockv _ [] = ([], blockv)
train blockv fn (h : t) = do
    let (error, blockInt, blockDouble) = fn blockv h
        (errorRest, retBlock) = train (blockInt, blockDouble) fn t
        error' = P.sum (toList error)
    (error' : errorRest, retBlock)

data Options = Options { optRandSeed :: Int, optTrueRand :: Bool, optLayers :: [LSpec], optOptimizer :: Optim, optCost :: CostFnT, optRepeat :: Int, optInputAF :: String}

startOptions :: Options
startOptions = Options { optRandSeed = 100, optTrueRand = False, optLayers = [[Relu 64], [Relu 64]], optOptimizer = (SGD 0.0001), optCost = MSE , optRepeat = 10, optInputAF = "Relu" }

options :: [OptDescr (Options -> IO Options)]
options = [ Option "s" ["seed"] (ReqArg (\arg -> \opt -> return opt { optRandSeed = (read arg :: Int) }) "100") "random seed",
            Option "r" ["rand"] (NoArg (\opt -> return opt { optTrueRand = True})) "use truly random seed",
            Option "l" ["layers"] (ReqArg (\arg -> \opt -> return opt { optLayers = (read arg :: [LSpec]) }) "[[Sigmoid 2], [Sigmoid 3], [Sigmoid 1]]") "specify layers",
            Option "O" ["optim"] (ReqArg (\arg -> \opt -> return opt { optOptimizer = (read arg :: Optim) }) "SGD 0.0001") "specify optimizer",
            Option "c" ["cost"] (ReqArg (\arg -> \opt -> return opt { optCost = (read arg :: CostFnT) }) "MSE" ) "specify cost function." ,
            Option "R" ["repeat"] (ReqArg (\arg -> \opt -> return opt { optRepeat = (read arg :: Int)}) "10" ) "specify number of epochs" ,
            Option "I" ["input-af"] (ReqArg (\arg -> \opt -> return opt { optInputAF = arg }) "Relu") "specify input activation function" ]

getSeed :: Int -> Bool -> IO StdGen
getSeed x False = return (mkStdGen x)
getSeed _ True = getStdGen

getAF :: String -> ActFunc
getAF "Sigmoid" = Sigmoid (28*28)
getAF "Relu" = Relu (28*28)
getAF "Ident" = Ident (28*28)
getAF "Softmax" = Softmax (28*28)
getAF "TanH" = TanH (28*28)


main :: IO ()
main = do
    hSetBuffering stdout LineBuffering
    mnistImages <- BS.readFile "train-images-idx3-ubyte"
    mnistLabels <- BS.readFile "train-labels-idx1-ubyte"
    args <- getArgs
    let (actions, _, errors) = getOpt RequireOrder options args
    opts <- P.foldl (>>=) (return startOptions) actions
    let Options { optRandSeed = seed, optTrueRand = trueRand, optLayers = layers, optOptimizer = optim, optCost = costFnT , optRepeat = numEpochs , optInputAF = af } = opts
    g <- getSeed seed trueRand
    let n = mkNetwork g ([[(getAF af)]] P.++ layers P.++ [[Softmax 10]]) optim 
        mnistImages' = BS.drop 16 mnistImages 
        mnistLabels' = BS.drop 8 mnistLabels
        labelVects = getLabels mnistLabels'
        imageVects = getImages mnistImages' `using` parBuffer numCapabilities rdeepseq
        zipped = P.zip imageVects labelVects
        repeated = P.take numEpochs (P.repeat zipped)
        folded = P.foldr (P.++) [] repeated
        (blinfo, blockAV) = network2block n
        blockV = run blockAV
        fn x y = trainOnce (ANN blinfo x) (costFn costFnT) y
        fn' = runN fn 
        (errors, output) = train blockV fn' folded
        errorsStr = P.map (\x -> (printf "%.5F" x ) P.++ "\n") errors
        bsout = block2bs (blinfo, output)
    P.putStr (P.foldr (P.++) "" errorsStr)
    BS.writeFile "mnist.ann" (toStrict bsout)

