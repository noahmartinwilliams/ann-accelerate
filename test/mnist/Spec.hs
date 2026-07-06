module Main where

import Control.DeepSeq
import Control.Monad.Reader
import Control.Monad.State
import Control.Parallel.Strategies
import Data.Array.Accelerate as A
import qualified Data.ByteString as B
import Data.List.Split
import Data.Maybe
import ML.ANN.Types
import ML.ANN.Network
import Neural
import Prelude as P
import System.Directory
import System.IO
import System.Random
import Tester
import Types

main :: IO ()
main = do
    createDirectoryIfMissing False "/tmp/results/"
    c <- readFile "configsMnist.txt"
    tri <- B.readFile "train-images.idx3-ubyte"
    tra <- B.readFile "train-labels.idx1-ubyte"
    ti <- B.readFile "t10k-images-idx3-ubyte"
    ta <- B.readFile "t10k-labels-idx1-ubyte"
    let lines = endBy "\n" c
        confs = P.map getConf lines
        filtered = P.filter (isJust) confs
        (frist: unJusted) = P.map (\(Just x) -> x) filtered
        nums = [0..]
        seed = mkStdGen 100
    runner (SampFiles { testImgs = B.drop 16 ti, testAnswers = B.drop 8 ta, trainImgs = B.drop 16 tri, trainAnswers = B.drop 8 tra}) (defaultState seed frist) (P.zip nums unJusted)  (A.fromList (Z:.1) [1], A.fromList (Z:.1) [1.0])


openFileS :: St -> IO St
openFileS s@(St { stFileToOpen = "" }) = return s
openFileS s@(St { stFileToOpen = fname }) = do
    handle <- openFile fname WriteMode 
    hSetBuffering handle (BlockBuffering Nothing)
    return (s { stOpenFile = handle, stFileToOpen = "" })

doIO :: St -> String -> IO St
doIO st str = do
    if (stCloseFile st)
    then do
        writeFileS st str
        hClose (stOpenFile st)
        s <- openFileS st
        return (s { stFileToOpen = "" , stFileToWrite = "", stCloseFile = False})
    else if ((stFileToOpen st) P.== "")
    then do
        writeFileS st str
        return st
    else do
        openFileS st

writeFileS :: St -> String -> IO ()
writeFileS s@(St { stOpenFile = h}) str = do
    hPutStr h str

runner :: SampFiles -> St -> [(Int, Conf)] -> (Vector Int, Vector Double) -> IO ()
runner _ _ [] _ = return ()
runner sf st ((i, c) : r) block = do
    let g = mkStdGen 100
        ((str, vi, vd), st') = runState (runReaderT (runner' block g sf i) c ) st
    st'' <- doIO st' str
    if (stTestPhase st'') P.&& ((stTestImgs st'') P.== []) P.&& ((stNumEpochs st'') P.== 0)
    then do
        runner sf (st'' {stNumEpochs = (stNumEpochs st'') - 1}) ((i, c) : r) block
    else if (stTestPhase st'') P.&& ((stTestImgs st'') P.== [])
    then
        runner sf (defaultState g c) r block
    else do
        runner sf st'' ((i, c) : r) (vi, vd)


runner' :: (Vector Int, Vector Double) -> StdGen -> SampFiles -> Int -> Mon (String, Vector Int, Vector Double)
runner' (blockArgI, blockArgD) g sf i = do
    let errsFile = "/tmp/results/errs-" P.++ (show i) P.++ ".txt" 
        testFile = "/tmp/results/test-" P.++ (show i) P.++ ".txt"
    (trainer, trainSamps, (blockI, blockD), testSamps) <- mkNeuralFns g i sf
    ti <- gets stTrainImgs
    started <- gets stStart
    testPhase <- gets stTestPhase
    if (ti P.== []) P.&& started
    then do
        modify (\s -> s { stCloseFile = False, stFileToWrite = "", stFileToOpen = errsFile })
        modify (\s -> s { stStart = False, stTrainFn = Just trainer, stTrainImgs = trainSamps, stTestImgs = testSamps, stTestPhase = False})
        return ("", blockI, blockD)
    else if (ti P.== []) P.&& (testPhase P.== False)
    then do
        modify (\s -> s {stFileToOpen = testFile, stCloseFile = True, stTestPhase = True})
        return ("", blockArgI, blockArgD)
    else if testPhase
    then do
        mkTester (blockArgI, blockArgD)
        runTester (blockArgI, blockArgD)
    else do
        runTrainer (blockArgI, blockArgD)

