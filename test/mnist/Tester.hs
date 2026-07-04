module Tester where

import Control.Monad.State
import Data.Array.Accelerate as A
import Data.Array.Accelerate.LLVM.PTX as PTX
import ML.ANN.Block
import ML.ANN.Network
import Prelude as P
import Text.Printf
import Types

mkTester :: (Vector Int, Vector Double) -> Mon ()
mkTester block = do
    jblinfo <- gets stBLInfo
    let (Just blinfo) = jblinfo
        testFn = PTX.runN (\x -> inferNetwork (block2network blinfo (use block)) (A.transpose x))
    modify (\s -> s { stTestFn = Just testFn})

runTester :: (Vector Int, Vector Double) -> Mon (String, Vector Int, Vector Double)
runTester (vi, vd) = do
    testImgs <- gets stTestImgs
    jfn <- gets stTestFn
    let (Just fn) = jfn
    if testImgs P.== []
    then do
        modify (\s -> s { stCloseFile = True})
        return ("", vi, vd)
    else do
        let ((fi, fa) : r) = testImgs
            result = fn fi 
            result' = A.toList result
            fa' = A.toList fa
            err = P.zipWith (\x -> \y -> (x - y) * (x - y)) result' fa'
            err' = P.sum err
            err'' = (printf "%.5f" err') P.++ "\n"
        modify (\s -> s { stTestImgs = r})
        return (err'', vi, vd)

