module Neural where

import Control.Monad.Reader
import Control.Monad.State
import Data.Array.Accelerate as A
import Data.Array.Accelerate.LLVM.PTX as PTX
import ML.ANN.Block
import ML.ANN.ErrorFn
import ML.ANN.Network
import ML.ANN.Types
import Prelude as P
import Samps
import System.Random
import Text.Printf
import Types

mkNeuralFns :: StdGen -> Int -> SampFiles -> Mon (TrainFn, [(Matrix Double, Matrix Double)], (Vector Int, Vector Double), [(Matrix Double, Matrix Double)] )
mkNeuralFns seed runNo sampFiles = do
    errfn <- reader costF
    opt <- reader optimizer
    mbs <- reader miniBatchSize
    let errFn' = getErrorFn errfn
    neural <- getNeural seed opt
    let (blinfo, block) = network2block neural
        block' = PTX.run block
    let trainFn = PTX.runN (\x -> \y -> trainMiniBatch mbs blinfo x (prepareSamp y))
        trainSamps = getSamps mbs (trainImgs sampFiles) (trainAnswers sampFiles)
        testSamps = getSamps 1 (testImgs sampFiles) (testAnswers sampFiles)
    modify (\s -> s { stBLInfo = Just blinfo})
    return (trainFn, trainSamps, PTX.run block, testSamps)

    
getErrorFn :: String -> ErrorFn
getErrorFn "MSE" = (mseErrorFn, dmseErrorFn)
getErrorFn "CrossEntropy" = (crossEntropyErrorFn, dcrossEntropyErrorFn)

getNeural :: StdGen -> String -> Mon Network
getNeural g "Adam" = do
    cnf <- ask
    lr1 <- reader lr
    b1 <- reader beta1
    b2 <- reader beta2
    mbs <- reader miniBatchSize
    let errFn = getErrorFn (costF cnf)
        lsp = read (layers cnf) :: [LSpec]
        iaf = read (inputAF cnf) :: ActFunc
        net = mkNetwork g (([((28*28), iaf)] : lsp) P.++ [[(10, SoftMax)]]) (AdamOptim (constant lr1) (constant b1) (constant b2)) errFn
    return net

getNeural g "SGD" = do
    cnf <- ask
    lr1 <- reader lr
    errFn <- reader costF
    mbs <- reader miniBatchSize
    let lsp = read (layers cnf) :: [LSpec]
        errFn' = getErrorFn errFn
        iaf = read (inputAF cnf) :: ActFunc
        net = mkNetwork g (([((28*28), iaf)] : lsp) P.++ [[(10, SoftMax)]]) (SGDOptim (constant lr1)) errFn'
    return net


runTrainer ::  (Vector Int, Vector Double) -> Mon (String, Vector Int, Vector Double)
runTrainer block = do
    samp <- gets stTrainImgs
    jtfn <- gets stTrainFn
    let (Just tfn) = jtfn
        (samp' : r) = samp
        (err, vi, vd) = tfn block samp'
        err' = P.map (\x -> printf "%.5f" x) (A.toList err)
        err'' = P.foldr (\x -> \y -> x P.++ "," P.++ y) "\n" err'
    modify (\s -> s { stTrainImgs = r, stFileToWrite = err''})
    return (err'', vi, vd)
