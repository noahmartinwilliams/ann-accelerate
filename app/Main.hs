module Main where

import ML.ANN
import System.Random
import Data.Array.Accelerate.Interpreter
import System.IO
import Data.Array.Accelerate as A
import Prelude as P
import Text.Printf

maybeNegativeMod :: Int -> Int -> Int -> Int
maybeNegativeMod inp sign modulus = do
    let inp2 = inp `P.mod` modulus
        inp3 = if (sign `P.mod` 2) P.== 0 then inp2 else -inp2
    inp3

genSamples :: [Int] -> [(Double, Double, Double)]
genSamples ( h1 : h2 : h3 : h4 : rest ) = do
    let h1' = maybeNegativeMod h1 h2 1000
        h2' = maybeNegativeMod h3 h4 1000
        h1_2 = P.fromIntegral h1' :: Double
        h2_2 = P.fromIntegral h2' :: Double
        result = if ((h1_2 P.>= 0.0) P.|| (h2_2 P.>= 0.0)) then 1.0 else 0.0
    ((h1_2, h2_2, result) : (genSamples rest))

train :: BlockV -> (BlockV -> (Vector Double, Vector Double) -> (Vector Double, Vector Int, Vector Double)) -> [(Double, Double, Double)] -> ([Vector Double], BlockV)
train blockV _ [] = ([], blockV)
train (blockI, blockD) fn ((i1, i2, o) : rest) = do
    let input = fromList (Z:.2) [i1, i2]
        output = fromList (Z:.1) [o]
        (err, output2I, output2D) = (fn (blockI, blockD) (input, output))
        (errRest, output3) = train (output2I, output2D) fn rest
    (err : errRest, output3)

sample2vect :: (Double, Double, Double) -> Acc (Matrix Double)
sample2vect (a, b, _) = use (fromList (Z:.2:.1) [a, b])

main :: IO ()
main = do
    let g = mkStdGen 100
    let n = mkNetwork g [[Relu 2], [Relu 5], [Sigmoid 1]] (SGD 0.0005)
        integers = randoms g :: [Int]
        samples = genSamples integers
        (blinfo, blockOut) = network2block n
        fn = runN (\a -> \b -> trainOnce (ANN blinfo a) mseCFn b)
        (errors, output) = train (run blockOut) fn (P.take 100000 samples)
        errorsLists = P.map (\x -> (toList x) ) errors
        errorsStrs = P.map (\[x] -> printf "%.5F" x) errorsLists
    hSetBuffering stdout LineBuffering
    putStr (P.foldr (\x -> \y -> x P.++ "\n" P.++ y) "" errorsStrs)
