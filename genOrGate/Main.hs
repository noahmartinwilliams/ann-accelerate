module Main where

import Prelude as P
import System.Random
import System.IO
import Text.Printf
import Control.Parallel.Strategies
import GHC.Conc

maybeNegativeMod :: Int -> Int -> Int -> Int
maybeNegativeMod inp sign modulus = do
    let inp2 = inp `P.mod` modulus
        inp3 = if (sign `P.mod` 2) P.== 0 then inp2 else -inp2
    inp3

genSamples :: [Int] -> [(Double, Double, Double)]
genSamples ( h1 : h2 : h3 : h4 : rest ) | (((P.abs h1) `P.mod` 1000) P.> 100) P.&& (((P.abs h3) `P.mod` 1000) P.> 100) = do
    let h1' = maybeNegativeMod h1 h2 1000
        h2' = maybeNegativeMod h3 h4 1000
        h1_2 = (P.fromIntegral h1' :: Double) / 1000.0
        h2_2 = (P.fromIntegral h2' :: Double) / 1000.0
        result = if ((h1_2 P.>= 0.0) P.|| (h2_2 P.>= 0.0)) then 1.0 else 0.0
    ((h1_2, h2_2, result) : (genSamples rest))
genSamples ( _ : _ : _ : _ : rest) = genSamples rest

main :: IO ()
main = do
    hSetBuffering stdout LineBuffering 
    let g = mkStdGen 100
        toTake = 10000000
        integers = randoms g :: [Int]
        samples = genSamples integers
        sampleLines = (P.map (\(x, y, z) -> (printf "%.5F" x) P.++ "," P.++ (printf "%.5F" y) P.++ "#" P.++ (printf "%.5F" z) P.++ "\n") samples) 
        samplesLinesOut = (P.take toTake sampleLines) -- `using` parListChunk numCapabilities rdeepseq
    putStr (P.foldr (P.++) "" samplesLinesOut )
