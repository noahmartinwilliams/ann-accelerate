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

charToList :: Int -> [Double]
charToList 0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
charToList 1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
charToList 2 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
charToList 3 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
charToList 4 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
charToList 5 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
charToList 6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
charToList 7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
charToList 8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
charToList 9 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

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

main :: IO ()
main = do
    hSetBuffering stdout LineBuffering
    mnistImages <- BS.readFile "mnist-dataset/train-images.idx3-ubyte"
    mnistLabels <- BS.readFile "mnist-dataset/train-labels.idx1-ubyte"
    let g = mkStdGen 200
        n = mkNetwork g [[Relu (28*28)], [Relu 32], [Relu 32], [Softmax 10]] (Adam 0.000001 0.9 0.999)
        --n = mkNetwork g [[Relu (28*28)], [Relu 16], [Softmax 10]] (SGD 0.000001)
        mnistImages' = BS.drop 16 mnistImages
        mnistLabels' = BS.drop 8 mnistLabels
        labelVects = getLabels mnistLabels'
        imageVects = getImages mnistImages'
        zipped = P.zip imageVects labelVects
        repeated = P.take 20 (P.repeat zipped)
        folded = P.foldr (P.++) [] repeated
        (blinfo, blockAV) = network2block n
        blockV = run blockAV
        fn x y = trainOnce (ANN blinfo x) crossEntropyCFn y
        fn' = runN fn 
        (errors, output) = train blockV fn' folded
        errorsStr = P.map (\x -> (printf "%.5F" x ) P.++ "\n") errors
        bsout = block2bs (blinfo, output)
    P.putStr (P.foldr (P.++) "" errorsStr)
    BS.writeFile "mnist.ann" (toStrict bsout)

