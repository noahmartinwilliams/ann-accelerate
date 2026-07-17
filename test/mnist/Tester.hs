module Tester where

import Control.Monad.Reader
import Control.Monad.State
import Data.Array.Accelerate as A
import Data.Array.Accelerate.LLVM.PTX as Dev
import Data.ByteString.Char8
import Data.List
import Misc
import ML.ANN.Block
import ML.ANN.Network
import Prelude as P
import Text.Printf
import Types

runTester :: Int -> TestFn -> Mon TestFn
runTester i tfn = do
    samps <- gets stTestSamps
    phase <- gets stPhase
    block <- gets stBlock
    blinfo <- gets stBLInfo
    toOpen <- gets stFilesToOpen
    let testFile = "/tmp/results/test-" P.++ (show i) P.++ ".txt"
    let tfn' = Dev.runN (inferNetwork (block2network blinfo (use block)))
    if phase P.== Test1
    then do
        modify (\s -> s { stPhase = Test2, stFilesToOpen = testFile : toOpen })
        return tfn'
    else if (samps P./= []) P.&& (phase P.== Test2)
    then do
        let (s1 : sr) = samps 
            res = tfn (P.fst s1)
            corrAns = P.snd s1
            resL = A.toList res
            corrAnsL = A.toList corrAns
            res' = P.zipWith (\x -> \y -> (x -y) * (x - y)) resL corrAnsL
            isa = isRightAnswer (P.zip resL corrAnsL)
            res'' = printf "%.5f\n" (P.sum (res'))
        modify (\s -> s { stTestSamps = sr })
        writer testFile (pack ((show isa) P.++ "," P.++ res''))
        return tfn
    else if phase P.== Test2 
    then do
        closer testFile
        modify (\s -> s { stPhase = Save })
        return tfn
    else
        return tfn


isRightAnswer :: [(Double, Double)] -> Int
isRightAnswer ls = do
    let sorted = Data.List.sort ls
        ((_, ans) : _ ) = sorted
    if ans P.> 0.0
    then
        1
    else
        0
