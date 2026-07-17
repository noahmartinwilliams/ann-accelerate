{-# LANGUAGE BangPatterns #-}
module Trainer where

import Control.DeepSeq
import Control.Monad.Reader
import Control.Monad.State
import Data.Array.Accelerate as A
import Data.Array.Accelerate.LLVM.PTX as Dev
import Data.ByteString.Char8 as BS
import Misc
import ML.ANN.Network
import Prelude as P
import Text.Printf
import Types

runTrainer :: Int -> TrainFn -> Mon TrainFn
runTrainer i tfn = do
    samps <- gets stTrainSamps
    phase <- gets stPhase
    blinfo <- gets stBLInfo
    mbs <- reader miniBatchSize
    toOpen <- gets stFilesToOpen
    toClose <- gets stFilesToClose
    block <- gets stBlock
    openFiles <- gets stFilesToWrite
    let trainerFile = "/tmp/results/errs-" P.++ (show i) P.++ ".txt"
    if phase P.== Start 
    then do
        let trainer = Dev.runN (trainMiniBatch mbs blinfo)
        modify (\s -> s { stPhase = Train, stFilesToOpen = trainerFile : toOpen})
        return trainer
    else if (phase P.== Train) P.&& (samps P.== [])
    then do
        closer trainerFile
        modify (\s -> s { stPhase = Test1})
        return tfn
    else if phase P.== Train P.&& (samps P./= [])
    then do
        let (samp1 : sampsr) = samps 
            (res1, block') = tfn block samp1
            res1' = (err2Str res1)
        writer trainerFile (BS.pack res1')
        --writer trainerFile (BS.pack ((show block') P.++ "\n"))
        modify (\s -> s { stBlock = block', stTrainSamps = sampsr})
        return tfn
    else
        return tfn
            

err2Str :: Vector Double -> String
err2Str v = do
    let l = A.toList v
        s = P.sum l
        strs = P.map (\x -> (printf "%.5f" x) P.++ ",") (s : l)
    P.foldr (P.++) "\n" strs

