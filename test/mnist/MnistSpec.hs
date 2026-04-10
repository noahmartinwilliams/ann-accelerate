{-# LANGUAGE DeriveGeneric, OverloadedStrings #-}
module Main where

import Conf
import Control.Monad.Reader
import Data.Aeson 
import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.LLVM.PTX as PTX
import qualified Data.ByteString as B
import Data.List.Split
import Data.Maybe
import GHC.Generics
import ML.ANN.Block
import ML.ANN.ErrorFn
import ML.ANN.Network
import ML.ANN.Types
import Neural
import Prelude as P
import System.Directory
import System.IO
import System.Random

main :: IO ()
main = do
    createDirectoryIfMissing False "/tmp/results"
    hSetBuffering stdout (BlockBuffering (Just 512))
    c <- readFile "configsMnist.txt"
    let lines = endBy "\n" c
        g = 100
    trainImgs <- B.readFile "train-images.idx3-ubyte"
    trainAnswers <- B.readFile "train-labels.idx1-ubyte"
    testImgs <- B.readFile "t10k-images-idx3-ubyte"
    testAnswers <- B.readFile "t10k-labels-idx1-ubyte"
    let configs = P.filter (isJust) (P.map getConf lines)
        configs' = P.map (\(Just x) -> x) configs
        fnums = [0..]
        strs = P.zipWith (\x -> \y -> runReader (runNeural g x trainImgs trainAnswers testImgs testAnswers ) y) fnums configs' 
    saveResults strs


saveResults :: [(String, String, String, String)] -> IO ()
saveResults [] = return ()
saveResults ((fname, fcontents, fname', fcontents') : tail) = do
    writeFile fname fcontents
    writeFile fname' fcontents'
    saveResults tail

