module Main where

import Prelude as P
import Data.Array.Accelerate as A
import Data.Array.Accelerate.LLVM.PTX
import ML.ANN

import System.IO
import Data.ByteString as BS
import Data.List.Split
import Data.List
import Data.Char
import Text.Printf
import System.Environment


getLabels :: BS.ByteString -> [Int]
getLabels bs = do
    let unpacked = BS.unpack bs
        characters = P.map (\x -> P.fromIntegral x :: Int ) unpacked
    characters

getImages :: BS.ByteString -> [Vector Double]
getImages bs = do
    let unpacked = BS.unpack bs
        integers = P.map (\x -> P.fromIntegral x :: Int) unpacked
        doubles = P.map (\x -> P.fromIntegral x :: Double) integers
        adjusted = P.map (\x -> (x - 128.0) / 128.0) doubles
        chunked = chunksOf (28*28) adjusted
        vectored = P.map (\x -> fromList (Z:.(28*28)) x) chunked
    vectored

go :: (Vector Double -> Vector Double) -> [(Vector Double, Int)] -> [Bool]
go _ [] = []
go fn ( ( input, expected ) : rest ) = do
    let actual = fn input
        ints = P.take 10 [0..]
        list = P.zip (toList actual) ints
        sorted = Data.List.sort list
        ((_, answer) : _) = Data.List.reverse sorted
    if answer P.== expected then True : (go fn rest) else False : (go fn rest)

writer :: [String] -> IO ()
writer [] = return ()
writer ( head : tail) = do
    System.IO.putStr head
    writer tail

main :: IO ()
main = do
    hSetBuffering stdout LineBuffering
    mnistImages <- BS.readFile "t10k-images-idx3-ubyte"
    mnistLabels <- BS.readFile "t10k-labels-idx1-ubyte"
    args <- getArgs
    annFD <- BS.readFile (args P.!! 0)
    (blinfo, block) <- bs2block (fromStrict annFD)
    let fn x = calcNetwork (block2network (blinfo, (use block))) (normalize x)
        fn2 = runN fn
        mnistImages' = BS.drop 16 mnistImages
        mnistLabels' = BS.drop 8 mnistLabels
        labelVects = getLabels mnistLabels'
        imageVects = getImages mnistImages'
        outputs = go fn2 (P.zip imageVects labelVects)
        outputsStrs = P.map (\x -> (show x) P.++ "\n") outputs
    writer outputsStrs

