module Main where

import ML.ANN
import System.Random
import Data.Array.Accelerate.Interpreter
import System.IO
import Data.Array.Accelerate as A
import Prelude as P
import Text.Printf

genSamples :: [Int] -> [(Double, Double, Double)]
genSamples ( h1 : h2 : rest ) = do
    let h1' = (P.abs h1) 
        h2' = (P.abs h2) 
        i1 = if h1' P.>= 500 then 500 + (h1' `P.mod` 500) else h1' `P.mod` 500
        i2 = if h2' P.>= 500 then 500 + (h2' `P.mod` 500) else h2' `P.mod` 500
        i1' = (P.fromIntegral i1 :: Double) / 1000.0
        i2' = (P.fromIntegral i2 :: Double) / 1000.0
        output = if (h1 P.>= 500) P.|| (h2 P.>= 500) then 1.0 else 0.0
    (i1', i2', output) : (genSamples rest)

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
    g <- getStdGen
    let n = mkNetwork g [[Relu 2], [Relu 4], [Relu 1]] (SGD (constant 0.0001))
        integers = randoms g :: [Int]
        samples = genSamples integers
        (blinfo, blockOut) = network2block n
        fn = runN (\a -> \b -> trainOnceLinReg (ANN blinfo a) b)
        (errors, output) = train (run blockOut) fn (P.take 100000 samples)
        errorsLists = P.map (\x -> (toList x) ) errors
        errorsStrs = P.map (\[x] -> printf "%.5F" x) errorsLists
    hSetBuffering stdout LineBuffering
    putStr (P.foldr (\x -> \y -> x P.++ "\n" P.++ y) "" errorsStrs)
