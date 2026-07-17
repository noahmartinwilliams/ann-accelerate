module Main where

import Control.Monad.Reader
import Control.Monad.State
import Data.Aeson
import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter
import Data.ByteString as BS
import Data.List.Split
import Data.Map
import Data.Maybe
import ML.ANN.Block
import ML.ANN.Network
import ML.ANN.Types
import Neural
import Prelude as P
import Samps
import Saver
import SideEff
import System.Directory
import System.Random
import Tester
import Trainer
import Types

main :: IO ()
main = do
    createDirectoryIfMissing False "/tmp/results/"
    h1 <- BS.readFile "train-images.idx3-ubyte"
    h2 <- BS.readFile "train-labels.idx1-ubyte"
    h3 <- BS.readFile "t10k-images-idx3-ubyte"
    h4 <- BS.readFile "t10k-labels-idx1-ubyte"
    c <- P.readFile "configsMnist.txt"
    let lines = endBy "\n" c
    let configs = P.map (\x -> Data.Aeson.decode (fromString x)) lines
        filtered = P.filter (isJust) configs
        configs' = P.map (\(Just x) -> x ) filtered
        nums = [0..]
    go configs' nums (h1, h2, h3, h4)


defTrain :: BLInfo -> Acc (Vector Int, Vector Double) -> Acc (Matrix Double, Matrix Double) -> Acc (Vector Double, (Vector Int, Vector Double))
defTrain blinfo block samp = do
    let res = trainOnce blinfo block samp
        (a, _, block') = A.unlift res :: (Acc (Matrix Double), Acc (Matrix Double), Acc (Vector Int, Vector Double))
    A.lift (A.flatten a, block')

defStuff :: (BLInfo, (Vector Int, Vector Double), TestFn, TrainFn)
defStuff = do
    let g = mkStdGen 100
        n = mkNetwork g [[(100, Sigmoid)], [(10, Sigmoid)], [(1, Sigmoid)]] (SGDOptim (constant 0.0001)) MSEErrorFn
        (blinfo, block) = (network2block n)
        testfn = runN (inferNetwork n)
        trainfn = runN (defTrain blinfo)
    (blinfo, run block, testfn, trainfn)
        
go :: [Conf] -> [Int] -> SampSource -> IO ()
go [] _ _ = return ()
go (hc : rc) (hi : ri) sampSources@(h1, h2, h3, h4) = do
    let (blinfo, block, testfn, trainfn) = defStuff
    runner' hc (St {stTestSamps = getSamps 1 (BS.drop 16 h3) (BS.drop 8 h4), stPhase = Start, stBLInfo = blinfo, stTrainSamps = getSamps (miniBatchSize hc) (BS.drop 16 h1) (BS.drop 8 h2), stBlock = block, stFilesToWrite = Data.Map.empty, stFilesToOpen = [], stFiles = Data.Map.empty, stFilesToClose = []}) trainfn testfn hi (numEpochs hc)
    go rc ri sampSources 

runner :: Int -> TrainFn -> TestFn -> Mon (Bool, TrainFn, TestFn)
runner num train test = do
    phase <- gets stPhase
    if phase P.== Start
    then do
        getNeural (mkStdGen 100) 
        train' <- runTrainer num train
        test' <- runTester num test
        runSaver num
        return (False, train', test')
    else do
        train' <- runTrainer num train
        test' <- runTester num test
        runSaver num
        phase <- gets stPhase
        if phase P.== Done
        then
            return (True, train', test')
        else
            return (False, train', test')
    
runner' :: Conf -> St -> TrainFn -> TestFn -> Int -> Int -> IO ()
runner' c st train test num epochNum = do
    let ((finished, train', test'), st') = runState (runReaderT (runner num train test) c) st
    st'' <- doIO  st'
    if finished P.&& (epochNum P.== 0)
    then
        return ()
    else if finished 
    then
        runner' c st'' train' test' num (epochNum - 1)
    else
        runner' c st'' train' test' num epochNum
